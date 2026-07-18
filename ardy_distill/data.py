"""Bounded safetensors shards for exact teacher windows."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset, Sampler

from .teacher import TeacherWindowTrace


SHARD_FIELDS = (
    "initial_noise",
    "denoising_states",
    "clean_generation",
    "history_hybrid",
    "path_condition",
    "first_heading",
    "has_history",
    "encoder_body",
    "encoder_valid",
    "decoder_latent",
    "decoder_global_root",
    "decoder_local_root",
    "decoder_token_valid",
    "target_body",
    "global_translation",
)

# V3 adds categorical provenance without duplicating the large cached text
# features in every teacher window.  Prompt ids index the separate dual
# Llama/Qwen feature table.  V2 shards synthesize zeros for these fields when
# requested, so the existing path-only corpus remains valid unconditional
# replay rather than becoming unusable.
OPTIONAL_SHARD_FIELDS = (
    "prompt_id",
    "control_mode_id",
    "rollout_depth",
    "prompt_switch",
    "heading_condition",
)
SUPPORTED_SHARD_FIELDS = SHARD_FIELDS + OPTIONAL_SHARD_FIELDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TeacherShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        shard_size: int,
        storage_dtype: torch.dtype = torch.float16,
        prefix: str = "teacher",
        extra_fields: tuple[str, ...] = (),
        manifest_metadata: dict | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.storage_dtype = storage_dtype
        self.prefix = prefix
        self.extra_fields = tuple(extra_fields)
        unknown = sorted(set(self.extra_fields) - set(OPTIONAL_SHARD_FIELDS))
        if unknown:
            raise ValueError(f"unknown optional teacher shard fields: {unknown}")
        if len(set(self.extra_fields)) != len(self.extra_fields):
            raise ValueError("duplicate optional teacher shard fields")
        self.fields = SHARD_FIELDS + self.extra_fields
        self.manifest_metadata = dict(manifest_metadata or {})
        self._buffers: dict[str, list[torch.Tensor]] = {name: [] for name in self.fields}
        self._buffered = 0
        self._written = 0
        self._shard_index = 0
        self.records: list[dict] = []

    @property
    def written(self) -> int:
        return self._written

    @property
    def buffered(self) -> int:
        return self._buffered

    def add(
        self,
        trace: TeacherWindowTrace,
        extra_tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        batch = trace.initial_noise.shape[0]
        extra_tensors = {} if extra_tensors is None else extra_tensors
        if set(extra_tensors) != set(self.extra_fields):
            raise ValueError(
                "extra tensor fields must exactly match writer extra_fields: "
                f"got {sorted(extra_tensors)}, expected {sorted(self.extra_fields)}"
            )
        for field in SHARD_FIELDS:
            tensor = getattr(trace, field).detach().cpu().contiguous()
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=self.storage_dtype)
            self._buffers[field].append(tensor)
        for field in self.extra_fields:
            tensor = extra_tensors[field].detach().cpu().contiguous()
            if tensor.ndim == 0 or tensor.shape[0] != batch:
                raise ValueError(
                    f"extra tensor {field} must have batch dimension {batch}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=self.storage_dtype)
            self._buffers[field].append(tensor)
        self._buffered += batch
        while self._buffered >= self.shard_size:
            self._flush(self.shard_size)

    def _flush(self, count: int) -> None:
        tensors: dict[str, torch.Tensor] = {}
        for field, chunks in self._buffers.items():
            joined = torch.cat(chunks, dim=0)
            tensors[field] = joined[:count].contiguous()
            remainder = joined[count:]
            self._buffers[field] = [remainder] if len(remainder) else []

        filename = f"{self.prefix}-{self._shard_index:05d}.safetensors"
        path = self.output_dir / filename
        save_file(
            tensors,
            path,
            metadata={
                "schema": (
                    "ardy_teacher_window_v3" if self.extra_fields else "ardy_teacher_window_v2"
                ),
                "count": str(count),
                "storage_dtype": str(self.storage_dtype).removeprefix("torch."),
            },
        )
        record = {
            "file": filename,
            "count": count,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        self.records.append(record)
        self._buffered -= count
        self._written += count
        self._shard_index += 1
        self._write_manifest()

    def close(self) -> None:
        if self._buffered:
            self._flush(self._buffered)
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "schema": (
                "ardy_teacher_window_v3" if self.extra_fields else "ardy_teacher_window_v2"
            ),
            "count": sum(record["count"] for record in self.records),
            "storage_dtype": str(self.storage_dtype).removeprefix("torch."),
            "fields": list(self.fields),
            "shards": self.records,
            **({"metadata": self.manifest_metadata} if self.manifest_metadata else {}),
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class TeacherShardDataset(Dataset[dict[str, torch.Tensor]]):
    """Map-style dataset with a small LRU of loaded safetensor shards."""

    def __init__(
        self,
        root: str | Path,
        cache_shards: int = 2,
        fields: tuple[str, ...] | None = None,
    ) -> None:
        self.root = Path(root)
        self.cache_shards = max(1, int(cache_shards))
        self.fields = tuple(SHARD_FIELDS if fields is None else fields)
        unknown = sorted(set(self.fields) - set(SUPPORTED_SHARD_FIELDS))
        if unknown:
            raise ValueError(f"unknown teacher shard fields: {unknown}")
        self.shards: list[tuple[Path, int]] = []
        self._available_fields: dict[Path, frozenset[str]] = {}
        for manifest_path in sorted(self.root.rglob("manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") not in {
                "ardy_teacher_window_v2",
                "ardy_teacher_window_v3",
            }:
                continue
            available = frozenset(manifest.get("fields", ()))
            if not set(SHARD_FIELDS).issubset(available):
                raise ValueError(f"teacher manifest is missing base fields: {manifest_path}")
            unsupported = sorted(available - set(SUPPORTED_SHARD_FIELDS))
            if unsupported:
                raise ValueError(
                    f"teacher manifest contains unsupported fields {unsupported}: {manifest_path}"
                )
            for record in manifest["shards"]:
                path = manifest_path.parent / record["file"]
                self.shards.append((path, int(record["count"])))
                self._available_fields[path] = available
        if not self.shards:
            raise FileNotFoundError(f"no teacher shards found under {self.root}")
        self.offsets = [0]
        for _, count in self.shards:
            self.offsets.append(self.offsets[-1] + count)
        self._cache: OrderedDict[Path, dict[str, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return self.offsets[-1]

    def _find_shard(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        low, high = 0, len(self.shards)
        while low < high:
            middle = (low + high) // 2
            if self.offsets[middle + 1] <= index:
                low = middle + 1
            else:
                high = middle
        return low, index - self.offsets[low]

    def _load(self, path: Path, count: int) -> dict[str, torch.Tensor]:
        cached = self._cache.pop(path, None)
        if cached is None:
            available = self._available_fields[path]
            readable = tuple(name for name in self.fields if name in available)
            if self.fields == SHARD_FIELDS and available == frozenset(SHARD_FIELDS):
                cached = load_file(path, device="cpu")
            else:
                with safe_open(path, framework="pt", device="cpu") as handle:
                    cached = {name: handle.get_tensor(name) for name in readable}
            for name in self.fields:
                if name not in cached:
                    if name not in OPTIONAL_SHARD_FIELDS:
                        raise KeyError(f"required field {name} missing from {path}")
                    if name == "heading_condition":
                        reference = cached.get("path_condition")
                        cached[name] = (
                            reference.new_zeros(count, 64, 3)
                            if reference is not None
                            else torch.zeros(count, 64, 3, dtype=torch.float16)
                        )
                    else:
                        dtype = torch.bool if name == "prompt_switch" else torch.int64
                        cached[name] = torch.zeros(count, dtype=dtype)
        self._cache[path] = cached
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index, local_index = self._find_shard(index)
        path, count = self.shards[shard_index]
        tensors = self._load(path, count)
        return {name: tensors[name][local_index] for name in self.fields}

    def iter_shards(self) -> Iterator[dict[str, torch.Tensor]]:
        for path, count in self.shards:
            yield self._load(path, count)


class ShardShuffleSampler(Sampler[int]):
    """Shuffle shard order and rows while retaining shard-local I/O locality.

    A global random permutation makes almost every adjacent sample open a
    different multi-megabyte safetensors shard.  This sampler preserves the
    same per-epoch randomization without turning decoder training into an I/O
    benchmark.  All distributed ranks use the same deterministic order and
    Accelerate assigns disjoint batches to ranks.
    """

    def __init__(self, dataset: TeacherShardDataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_order = torch.randperm(len(self.dataset.shards), generator=generator)
        for shard_tensor in shard_order:
            shard_index = int(shard_tensor)
            start = self.dataset.offsets[shard_index]
            count = self.dataset.offsets[shard_index + 1] - start
            local_order = torch.randperm(count, generator=generator)
            for local_tensor in local_order:
                yield start + int(local_tensor)

    def __len__(self) -> int:
        return len(self.dataset)

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "epoch": self.epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError(
                f"sampler seed mismatch: checkpoint={state['seed']} current={self.seed}"
            )
        self.epoch = int(state["epoch"])


class RandomTeacherMixtureDataset(Dataset[dict[str, torch.Tensor]]):
    """Step-driven stochastic replay mixture with independent shard caches.

    ``primary_probability`` is applied on every access.  The requested index is
    intentionally ignored after bounds checking; DataLoader/Accelerate RNG state
    therefore fully determines the sampled source and sample and is checkpointed
    with the rest of the training state.
    """

    def __init__(
        self,
        primary: TeacherShardDataset,
        replay: TeacherShardDataset,
        primary_probability: float = 0.5,
    ) -> None:
        if not 0.0 < primary_probability < 1.0:
            raise ValueError("primary_probability must be strictly between zero and one")
        self.primary = primary
        self.replay = replay
        self.primary_probability = float(primary_probability)

    def __len__(self) -> int:
        return max(len(self.primary), len(self.replay))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source = self.primary if torch.rand(()) < self.primary_probability else self.replay
        sampled_index = int(torch.randint(len(source), ()).item())
        return source[sampled_index]


class IndexedTeacherMixtureDataset(Dataset[dict[str, torch.Tensor]]):
    """Index-preserving view over primary and replay teacher corpora.

    Unlike :class:`RandomTeacherMixtureDataset`, this class never replaces a
    requested index with a second random lookup.  A batch-aware sampler can
    therefore keep adjacent requests inside one safetensors shard while still
    enforcing the desired source mixture.
    """

    def __init__(
        self,
        primary: TeacherShardDataset,
        replay: TeacherShardDataset,
    ) -> None:
        self.primary = primary
        self.replay = replay
        self.replay_offset = len(primary)

    def __len__(self) -> int:
        return len(self.primary) + len(self.replay)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < self.replay_offset:
            return self.primary[index]
        return self.replay[index - self.replay_offset]


class ShardMixtureSampler(Sampler[int]):
    """Batch-local shard shuffle for an oversampled primary/replay mixture.

    Source selection occurs once per ``chunk_size`` samples, which is normally
    the per-process DataLoader batch size.  Rows remain shuffled, but every
    chunk is drawn from one source's shard-local permutation.  This avoids the
    per-sample cross-shard seeks of ``RandomTeacherMixtureDataset`` while
    allowing a small on-policy corpus to be replayed at a fixed probability.
    """

    def __init__(
        self,
        dataset: IndexedTeacherMixtureDataset,
        *,
        primary_probability: float,
        chunk_size: int,
        seed: int,
        samples_per_epoch: int | None = None,
    ) -> None:
        if not 0.0 < primary_probability < 1.0:
            raise ValueError("primary_probability must be strictly between zero and one")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.dataset = dataset
        self.primary_probability = float(primary_probability)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        requested = (
            max(len(dataset.primary), len(dataset.replay))
            if samples_per_epoch is None
            else int(samples_per_epoch)
        )
        self.samples_per_epoch = requested - requested % self.chunk_size
        if self.samples_per_epoch < self.chunk_size:
            raise ValueError("samples_per_epoch must contain at least one complete chunk")
        self.epoch = 0

    @staticmethod
    def _shard_local_order(
        dataset: TeacherShardDataset,
        generator: torch.Generator,
    ) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        shard_order = torch.randperm(len(dataset.shards), generator=generator)
        for shard_tensor in shard_order:
            shard_index = int(shard_tensor)
            start = dataset.offsets[shard_index]
            count = dataset.offsets[shard_index + 1] - start
            chunks.append(start + torch.randperm(count, generator=generator))
        return torch.cat(chunks)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        source_orders = {
            "primary": self._shard_local_order(self.dataset.primary, generator),
            "replay": self._shard_local_order(self.dataset.replay, generator),
        }
        positions = {"primary": 0, "replay": 0}
        offsets = {"primary": 0, "replay": self.dataset.replay_offset}
        chunks = self.samples_per_epoch // self.chunk_size
        for _ in range(chunks):
            source = (
                "primary"
                if float(torch.rand((), generator=generator)) < self.primary_probability
                else "replay"
            )
            order = source_orders[source]
            position = positions[source]
            if position + self.chunk_size > len(order):
                order = self._shard_local_order(
                    self.dataset.primary if source == "primary" else self.dataset.replay,
                    generator,
                )
                source_orders[source] = order
                position = 0
            selected = order[position : position + self.chunk_size] + offsets[source]
            positions[source] = position + self.chunk_size
            for index in selected:
                yield int(index)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def state_dict(self) -> dict[str, int | float]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "chunk_size": self.chunk_size,
            "samples_per_epoch": self.samples_per_epoch,
            "primary_probability": self.primary_probability,
        }

    def load_state_dict(self, state: dict[str, int | float]) -> None:
        expected = {
            "seed": self.seed,
            "chunk_size": self.chunk_size,
            "samples_per_epoch": self.samples_per_epoch,
            "primary_probability": self.primary_probability,
        }
        for name, value in expected.items():
            if state[name] != value:
                raise ValueError(
                    f"mixture sampler {name} mismatch: checkpoint={state[name]} current={value}"
                )
        self.epoch = int(state["epoch"])
