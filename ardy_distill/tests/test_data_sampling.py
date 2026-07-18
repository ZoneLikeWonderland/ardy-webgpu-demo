from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import torch
from torch.utils.data import Dataset

from ardy_distill.data import (
    IndexedTeacherMixtureDataset,
    ShardMixtureSampler,
    TeacherShardDataset,
    TeacherShardWriter,
)


_TRACE_SHAPES = {
    "initial_noise": (10, 148),
    "denoising_states": (11, 10, 148),
    "clean_generation": (10, 148),
    "history_hybrid": (1, 148),
    "path_condition": (64, 3),
    "first_heading": (2,),
    "has_history": (1,),
    "encoder_body": (4, 325),
    "encoder_valid": (1,),
    "decoder_latent": (11, 128),
    "decoder_global_root": (44, 5),
    "decoder_local_root": (44, 4),
    "decoder_token_valid": (11,),
    "target_body": (44, 325),
    "global_translation": (3,),
}


def _fake_trace(batch: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            name: torch.zeros(batch, *shape, dtype=torch.float32)
            for name, shape in _TRACE_SHAPES.items()
        }
    )


class _DummyShardDataset(Dataset):
    def __init__(self, shard_sizes: list[int], source: int) -> None:
        self.shards = [(f"shard-{index}", size) for index, size in enumerate(shard_sizes)]
        self.offsets = [0]
        for size in shard_sizes:
            self.offsets.append(self.offsets[-1] + size)
        self.source = source

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"source": self.source, "index": index}


class ShardMixtureSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.primary = _DummyShardDataset([8, 8], source=0)
        self.replay = _DummyShardDataset([8, 8, 8], source=1)
        self.dataset = IndexedTeacherMixtureDataset(self.primary, self.replay)

    def make_sampler(self, seed: int = 17) -> ShardMixtureSampler:
        return ShardMixtureSampler(
            self.dataset,
            primary_probability=0.70,
            chunk_size=4,
            samples_per_epoch=4000,
            seed=seed,
        )

    def test_chunks_never_mix_sources(self) -> None:
        indices = list(self.make_sampler())
        offset = len(self.primary)
        chunks = [indices[index : index + 4] for index in range(0, len(indices), 4)]
        for chunk in chunks:
            sources = {index >= offset for index in chunk}
            self.assertEqual(len(sources), 1)
        primary_fraction = sum(chunk[0] < offset for chunk in chunks) / len(chunks)
        self.assertAlmostEqual(primary_fraction, 0.70, delta=0.05)

    def test_indexed_dataset_preserves_requested_source(self) -> None:
        self.assertEqual(self.dataset[3]["source"], 0)
        self.assertEqual(self.dataset[len(self.primary) + 3]["source"], 1)

    def test_state_restore_reproduces_next_epoch(self) -> None:
        first = self.make_sampler(seed=29)
        list(first)
        state = first.state_dict()
        expected = list(first)

        restored = self.make_sampler(seed=29)
        restored.load_state_dict(state)
        self.assertEqual(list(restored), expected)

    def test_state_mismatch_is_rejected(self) -> None:
        sampler = self.make_sampler()
        state = sampler.state_dict()
        state["chunk_size"] = 8
        with self.assertRaises(ValueError):
            sampler.load_state_dict(state)


class TeacherShardV3Test(unittest.TestCase):
    def test_v3_metadata_round_trip_and_v2_zero_defaults(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            v3 = root / "v3"
            writer = TeacherShardWriter(
                v3,
                shard_size=2,
                extra_fields=(
                    "prompt_id",
                    "control_mode_id",
                    "rollout_depth",
                    "prompt_switch",
                    "heading_condition",
                ),
            )
            writer.add(
                _fake_trace(),
                {
                    "prompt_id": torch.tensor([7, 9]),
                    "control_mode_id": torch.tensor([1, 4]),
                    "rollout_depth": torch.tensor([0, 5]),
                    "prompt_switch": torch.tensor([False, True]),
                    "heading_condition": torch.randn(2, 64, 3),
                },
            )
            writer.close()
            fields = (
                "prompt_id",
                "control_mode_id",
                "rollout_depth",
                "prompt_switch",
                "heading_condition",
            )
            dataset = TeacherShardDataset(v3, fields=fields)
            self.assertEqual(int(dataset[0]["prompt_id"]), 7)
            self.assertEqual(int(dataset[1]["control_mode_id"]), 4)
            self.assertTrue(bool(dataset[1]["prompt_switch"]))
            self.assertEqual(tuple(dataset[0]["heading_condition"].shape), (64, 3))

            v2 = root / "v2"
            legacy_writer = TeacherShardWriter(v2, shard_size=2)
            legacy_writer.add(_fake_trace())
            legacy_writer.close()
            legacy = TeacherShardDataset(v2, fields=fields)
            self.assertEqual(int(legacy[0]["prompt_id"]), 0)
            self.assertFalse(bool(legacy[0]["prompt_switch"]))
            self.assertEqual(int(torch.count_nonzero(legacy[0]["heading_condition"])), 0)


if __name__ == "__main__":
    unittest.main()
