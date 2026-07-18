"""Offline dual text features for teacher/student motion distillation."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


QWEN_HIDDEN_STATE_LAYERS = (9, 18, 27)
QWEN_HIDDEN_SIZE = 2560
QWEN_FEATURE_DIM = len(QWEN_HIDDEN_STATE_LAYERS) * QWEN_HIDDEN_SIZE
LLM2VEC_FEATURE_DIM = 4096


class PromptFeatureTable:
    """Small in-memory table indexed by the prompt ids stored in V3 shards."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_encoder: str | None = None,
    ) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != "ardy_prompt_features_v1":
            raise ValueError(f"unexpected prompt feature schema: {manifest_path}")
        if not self.manifest.get("complete"):
            raise ValueError(f"prompt feature table is incomplete: {manifest_path}")
        self.encoder = str(self.manifest["encoder"])
        if expected_encoder is not None and self.encoder != expected_encoder:
            raise ValueError(
                f"expected {expected_encoder} features, found {self.encoder}: {manifest_path}"
            )
        feature_chunks = []
        id_chunks = []
        for record in self.manifest["shards"]:
            tensors = load_file(self.root / record["file"], device="cpu")
            feature_chunks.append(tensors["features"])
            id_chunks.append(tensors["prompt_ids"].long())
        self.features = torch.cat(feature_chunks, dim=0).contiguous()
        prompt_ids = torch.cat(id_chunks, dim=0)
        expected_ids = torch.arange(len(prompt_ids), dtype=torch.long)
        if not torch.equal(prompt_ids, expected_ids):
            raise ValueError(f"prompt feature ids are not contiguous: {manifest_path}")
        expected_shape = (
            int(self.manifest["count"]),
            int(self.manifest["feature_dim"]),
        )
        if self.features.shape != expected_shape:
            raise ValueError(
                f"feature table shape {tuple(self.features.shape)} != {expected_shape}"
            )
        if not torch.isfinite(self.features).all():
            raise ValueError(f"non-finite prompt features: {manifest_path}")

    def __len__(self) -> int:
        return self.features.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[1]

    def to(
        self,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> "PromptFeatureTable":
        self.features = self.features.to(device=device, dtype=dtype).contiguous()
        return self

    def lookup(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        prompt_ids = prompt_ids.to(device=self.features.device, dtype=torch.long)
        if prompt_ids.numel() and (
            int(prompt_ids.min()) < 0 or int(prompt_ids.max()) >= len(self)
        ):
            raise IndexError("prompt id outside feature table")
        return self.features[prompt_ids]


def pool_qwen_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    layers: Sequence[int] = QWEN_HIDDEN_STATE_LAYERS,
) -> torch.Tensor:
    """Concatenate FLUX.2's three Qwen layers, then mean-pool valid tokens.

    FLUX.2 itself keeps the full token sequence.  ARDY's released LLM2Vec
    condition is a single pooled token, so the compact motion student uses the
    same one-token interface without PCA or low-rank compression.
    """

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    selected = [hidden_states[index] for index in layers]
    if any(tensor.ndim != 3 for tensor in selected):
        raise ValueError("Qwen hidden states must have shape [batch, sequence, hidden]")
    concatenated = torch.cat(selected, dim=-1)
    mask = attention_mask.to(device=concatenated.device, dtype=concatenated.dtype).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1)
    pooled = (concatenated * mask).sum(dim=1) / denominator
    if pooled.shape[-1] != QWEN_FEATURE_DIM:
        raise ValueError(
            f"expected pooled Qwen feature dim {QWEN_FEATURE_DIM}, got {pooled.shape[-1]}"
        )
    return pooled


def qwen_chat_inputs(
    tokenizer,
    prompts: Sequence[str],
    *,
    max_sequence_length: int,
) -> dict[str, torch.Tensor]:
    """Apply the exact non-thinking FLUX.2 user chat template."""

    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    return tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_sequence_length,
    )


class Flux2QwenFeatureEncoder:
    """Load only FLUX.2's Qwen3 backbone and emit full 7680-D features."""

    def __init__(
        self,
        flux2_root: str | Path,
        device: str | torch.device,
        *,
        max_sequence_length: int = 128,
    ) -> None:
        from transformers import AutoTokenizer, Qwen3Model

        self.root = Path(flux2_root).resolve()
        self.text_encoder_path = self.root / "text_encoder"
        self.tokenizer_path = self.root / "tokenizer"
        if not self.text_encoder_path.is_dir() or not self.tokenizer_path.is_dir():
            raise FileNotFoundError(
                f"FLUX.2 text_encoder/tokenizer missing under {self.root}"
            )
        self.device = torch.device(device)
        self.max_sequence_length = int(max_sequence_length)
        if self.max_sequence_length < 8:
            raise ValueError("max_sequence_length is implausibly small")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            local_files_only=True,
        )
        self.model = Qwen3Model.from_pretrained(
            self.text_encoder_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, prompts: Sequence[str]) -> torch.Tensor:
        if not prompts:
            return torch.empty(0, QWEN_FEATURE_DIM, dtype=torch.float32)
        inputs = qwen_chat_inputs(
            self.tokenizer,
            prompts,
            max_sequence_length=self.max_sequence_length,
        )
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        output = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
        pooled = pool_qwen_hidden_states(output.hidden_states, inputs["attention_mask"])
        empty = torch.tensor(
            [not prompt.strip() for prompt in prompts],
            device=pooled.device,
            dtype=torch.bool,
        )
        pooled[empty] = 0
        # Qwen inference stays in its native BF16.  Cache the pooled feature
        # directly as FP16 because that is the eventual WebGPU input dtype;
        # do not introduce a pointless FP32 transfer in between.
        return pooled.to(device="cpu", dtype=torch.float16)


class ArdyLlamaFeatureEncoder:
    """Released ARDY Llama-3-8B/LLM2Vec feature encoder."""

    def __init__(
        self,
        models_root: str | Path,
        device: str | torch.device,
    ) -> None:
        import os

        from ardy.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder

        self.models_root = Path(models_root).resolve()
        self.device = str(device)
        previous_cwd = Path.cwd()
        try:
            # The released adapter config names its base as
            # ``meta-llama/...``.  Resolving that relative name from this root
            # keeps the complete load offline without editing upstream files.
            os.chdir(self.models_root)
            self.encoder = LLM2VecEncoder(
                base_model_name_or_path=(
                    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
                ),
                peft_model_name_or_path=(
                    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
                ),
                dtype="bfloat16",
                llm_dim=LLM2VEC_FEATURE_DIM,
                device=self.device,
            )
        finally:
            os.chdir(previous_cwd)
        self.encoder.eval()

    @torch.inference_mode()
    def __call__(self, prompts: Sequence[str]) -> torch.Tensor:
        if not prompts:
            return torch.empty(0, LLM2VEC_FEATURE_DIM, dtype=torch.float32)
        encoded, _lengths = self.encoder(list(prompts))
        encoded = encoded.reshape(len(prompts), LLM2VEC_FEATURE_DIM).to(
            device="cpu", dtype=torch.float16
        )
        empty = torch.tensor([not prompt.strip() for prompt in prompts], dtype=torch.bool)
        encoded[empty] = 0
        return encoded
