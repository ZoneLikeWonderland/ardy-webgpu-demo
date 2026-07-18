"""Control-aware sampling for the final text/control ARDY corpus."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


CONTROL_MODES = (
    "none",
    "mouse_sparse",
    "mouse_dense",
    "keyboard_velocity",
    "keyboard_heading",
)
CONTROL_MODE_TO_ID = {name: index for index, name in enumerate(CONTROL_MODES)}
CONTROL_MODE_WEIGHTS = {
    "none": 0.20,
    "mouse_sparse": 0.25,
    "mouse_dense": 0.20,
    "keyboard_velocity": 0.25,
    "keyboard_heading": 0.10,
}


@dataclass(frozen=True)
class ControlTrajectory:
    mode: str
    frame_indices: torch.Tensor
    root_xz: torch.Tensor
    heading_angles: torch.Tensor | None

    @property
    def mode_id(self) -> int:
        return CONTROL_MODE_TO_ID[self.mode]


def load_prompt_bank(path: str | Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if [record["prompt_id"] for record in records] != list(range(len(records))):
        raise ValueError("prompt bank ids must be contiguous")
    return records


class PromptControlSampler:
    """Sample prompts and only their compatible UI control modes."""

    def __init__(
        self,
        records: list[dict],
        generator: torch.Generator,
        *,
        unconditional_probability: float = 0.10,
        official_probability: float = 0.10,
    ) -> None:
        if not 0 <= unconditional_probability < 1:
            raise ValueError("unconditional probability must be in [0, 1)")
        if not 0 <= official_probability < 1:
            raise ValueError("official probability must be in [0, 1)")
        if unconditional_probability + official_probability >= 1:
            raise ValueError("unconditional and official probabilities leave no generated prompts")
        self.records = records
        self.generator = generator
        self.unconditional_probability = float(unconditional_probability)
        self.official_probability = float(official_probability)
        self.official_ids = torch.tensor(
            [
                record["prompt_id"]
                for record in records
                if record["source"] == "official_ardy_preset"
            ],
            dtype=torch.long,
        )
        self.generated_ids = torch.tensor(
            [
                record["prompt_id"]
                for record in records
                if record["source"] == "template_v1"
            ],
            dtype=torch.long,
        )
        if not len(self.official_ids) or not len(self.generated_ids):
            raise ValueError("prompt bank must contain official and generated entries")

    def sample_prompt_ids(self, batch: int) -> torch.Tensor:
        selector = torch.rand(batch, generator=self.generator)
        generated_index = torch.randint(
            len(self.generated_ids), (batch,), generator=self.generator
        )
        result = self.generated_ids[generated_index]
        official = selector < (
            self.unconditional_probability + self.official_probability
        )
        official_index = torch.randint(
            len(self.official_ids), (batch,), generator=self.generator
        )
        result[official] = self.official_ids[official_index[official]]
        result[selector < self.unconditional_probability] = 0
        return result

    def sample_control_mode(self, prompt_id: int) -> str:
        record = self.records[prompt_id]
        allowed = tuple(record["control_modes"])
        weights = torch.tensor(
            [CONTROL_MODE_WEIGHTS[name] for name in allowed], dtype=torch.float64
        )
        index = int(torch.multinomial(weights, 1, generator=self.generator))
        return allowed[index]

    def sample_trajectory(
        self,
        prompt_id: int,
        mode: str,
        *,
        start_xz: torch.Tensor,
        start_heading: float,
        current_velocity_xz: torch.Tensor | None = None,
        history_frames: int = 0,
        frames: int = 64,
        fps: int = 20,
    ) -> ControlTrajectory:
        record = self.records[prompt_id]
        if mode not in record["control_modes"]:
            raise ValueError(
                f"control mode {mode} is incompatible with prompt family {record['family']}"
            )
        if start_xz.shape != (2,):
            raise ValueError("start_xz must have shape [2]")
        if not 0 <= history_frames < frames:
            raise ValueError("history_frames must be within the conditioning window")
        if mode == "none":
            return ControlTrajectory(
                mode=mode,
                frame_indices=torch.empty(0, dtype=torch.long),
                root_xz=torch.empty(0, 2, dtype=torch.float32),
                heading_angles=None,
            )

        current_index = max(0, history_frames - 1)
        available = frames - current_index - 1
        if available < 10:
            raise ValueError("conditioning window has fewer than ten future frames")
        speed = self._sample_speed(record)
        direction = self._sample_direction(record, start_heading)
        current_velocity_xz = (
            torch.zeros(2, dtype=torch.float32)
            if current_velocity_xz is None
            else current_velocity_xz.float().cpu()
        )

        if mode == "mouse_sparse":
            return self._mouse_sparse(
                record,
                start_xz.float().cpu(),
                current_index,
                available,
                speed,
                direction,
                fps,
            )
        if mode == "mouse_dense":
            return self._mouse_dense(
                record,
                start_xz.float().cpu(),
                current_index,
                available,
                speed,
                direction,
                fps,
            )
        if mode in {"keyboard_velocity", "keyboard_heading"}:
            return self._keyboard(
                mode,
                record,
                start_xz.float().cpu(),
                current_velocity_xz,
                current_index,
                available,
                speed,
                direction,
                fps,
            )
        raise AssertionError(mode)

    def _uniform(self) -> float:
        return float(torch.rand((), generator=self.generator))

    def _normal(self) -> float:
        return float(torch.randn((), generator=self.generator))

    def _sample_speed(self, record: dict) -> float:
        minimum = float(record["speed_min_mps"])
        maximum = float(record["speed_max_mps"])
        if record["control_profile"] == "stationary" and self._uniform() < 0.40:
            return 0.0
        if record["family"] == "start_stop" and self._uniform() < 0.25:
            return 0.0
        return minimum + self._uniform() * (maximum - minimum)

    def _sample_direction(self, record: dict, start_heading: float) -> float:
        family = record["family"]
        jitter = 0.18 * self._normal()
        if family == "backward":
            return start_heading + math.pi + jitter
        if family == "lateral":
            side = -1.0 if self._uniform() < 0.5 else 1.0
            return start_heading + side * math.pi / 2 + jitter
        if family in {"turning", "curved_route", "dance"}:
            side = -1.0 if self._uniform() < 0.5 else 1.0
            return start_heading + side * (0.25 + 0.55 * self._uniform()) + jitter
        if family == "unconditional":
            return -math.pi + 2 * math.pi * self._uniform()
        return start_heading + jitter

    def _turn_scale(self, record: dict) -> float:
        if record["family"] in {"turning", "curved_route", "dance"}:
            return 0.035
        if record["family"] == "lateral":
            return 0.010
        return 0.006

    def _mouse_sparse(
        self,
        record: dict,
        start_xz: torch.Tensor,
        current_index: int,
        available: int,
        speed: float,
        direction: float,
        fps: int,
    ) -> ControlTrajectory:
        selector = self._uniform()
        points = 1 if selector < 0.55 else 2 if selector < 0.85 else 3
        preferred = (20, 30, 40, 60)
        final_offset = preferred[int(self._uniform() * len(preferred))]
        final_offset = min(final_offset, available)
        offsets = torch.linspace(
            max(8, final_offset / points),
            final_offset,
            points,
        ).round().long()
        offsets = torch.unique(offsets).clamp(min=1, max=available)
        positions = []
        position = start_xz.clone()
        previous = 0
        turn_scale = 0.55 if record["family"] in {"turning", "curved_route"} else 0.20
        for offset in offsets.tolist():
            direction += turn_scale * self._normal()
            seconds = (offset - previous) / fps
            velocity = torch.tensor(
                [math.sin(direction), math.cos(direction)], dtype=torch.float32
            ) * speed
            position = position + seconds * velocity
            positions.append(position.clone())
            previous = offset
        return ControlTrajectory(
            mode="mouse_sparse",
            frame_indices=offsets + current_index,
            root_xz=torch.stack(positions),
            heading_angles=None,
        )

    def _mouse_dense(
        self,
        record: dict,
        start_xz: torch.Tensor,
        current_index: int,
        available: int,
        speed: float,
        direction: float,
        fps: int,
    ) -> ControlTrajectory:
        frame_indices = torch.arange(current_index + 1, current_index + available + 1)
        positions = []
        position = start_xz.clone()
        turn_scale = self._turn_scale(record)
        turn_bias = 0.0
        if record["family"] in {"turning", "curved_route"}:
            turn_bias = (-1.0 if self._uniform() < 0.5 else 1.0) * 0.018
        for _ in range(available):
            direction += turn_bias + turn_scale * self._normal()
            velocity = torch.tensor(
                [math.sin(direction), math.cos(direction)], dtype=torch.float32
            ) * speed
            position = position + velocity / fps
            positions.append(position.clone())
        return ControlTrajectory(
            mode="mouse_dense",
            frame_indices=frame_indices,
            root_xz=torch.stack(positions),
            heading_angles=None,
        )

    def _keyboard(
        self,
        mode: str,
        record: dict,
        start_xz: torch.Tensor,
        current_velocity: torch.Tensor,
        current_index: int,
        available: int,
        speed: float,
        direction: float,
        fps: int,
    ) -> ControlTrajectory:
        # Literal demo semantics: two-second linear velocity transition and a
        # target waypoint every ten frames.
        future_frames = min(int(2.0 * fps), available)
        target_velocity = torch.tensor(
            [math.sin(direction), math.cos(direction)], dtype=torch.float32
        ) * speed
        position = start_xz.clone()
        positions = []
        velocities = []
        for frame in range(future_frames):
            alpha = (frame + 1) / future_frames
            velocity = (1 - alpha) * current_velocity + alpha * target_velocity
            position = position + velocity / fps
            positions.append(position.clone())
            velocities.append(velocity.clone())
        offsets = torch.arange(10, future_frames + 1, 10, dtype=torch.long)
        selected_positions = torch.stack([positions[int(offset) - 1] for offset in offsets])
        headings = None
        if mode == "keyboard_heading":
            # Match scripts/interactive_demo/gen_constraints.py exactly.
            headings = torch.tensor(
                [
                    math.atan2(
                        float(velocities[int(offset) - 1][1]),
                        float(velocities[int(offset) - 1][0]),
                    )
                    for offset in offsets
                ],
                dtype=torch.float32,
            )
        return ControlTrajectory(
            mode=mode,
            frame_indices=offsets + current_index,
            root_xz=selected_positions,
            heading_angles=headings,
        )


def pack_heading_condition(
    observed_motion: torch.Tensor | None,
    motion_mask: torch.Tensor | None,
    *,
    frames: int = 64,
) -> torch.Tensor:
    """Pack normalized teacher heading cos/sin plus validity for the student."""

    if observed_motion is None or motion_mask is None:
        batch = 1 if observed_motion is None else observed_motion.shape[0]
        device = torch.device("cpu") if observed_motion is None else observed_motion.device
        dtype = torch.float32 if observed_motion is None else observed_motion.dtype
        return torch.zeros(batch, frames, 3, device=device, dtype=dtype)
    batch = observed_motion.shape[0]
    result = observed_motion.new_zeros(batch, frames, 3)
    count = min(frames, observed_motion.shape[1])
    result[:, :count, :2] = observed_motion[:, :count, 3:5]
    valid = (motion_mask[:, :count, 3] > 0) | (motion_mask[:, :count, 4] > 0)
    result[:, :count, 2] = valid.to(dtype=result.dtype)
    result[:, :count, :2] *= result[:, :count, 2:3]
    return result

