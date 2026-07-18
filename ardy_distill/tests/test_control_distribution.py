from __future__ import annotations

from collections import Counter

import torch

from ardy_distill.control_distribution import (
    CONTROL_MODES,
    PromptControlSampler,
    load_prompt_bank,
    pack_heading_condition,
)


def test_prompt_and_control_distribution_is_compatible_and_deterministic() -> None:
    records = load_prompt_bank("distill_data/text_control_v1/prompt_bank.jsonl")
    first = PromptControlSampler(records, torch.Generator().manual_seed(81))
    second = PromptControlSampler(records, torch.Generator().manual_seed(81))
    ids_a = first.sample_prompt_ids(10_000)
    ids_b = second.sample_prompt_ids(10_000)
    assert torch.equal(ids_a, ids_b)
    unconditional = float((ids_a == 0).float().mean())
    official = sum(records[int(index)]["source"] == "official_ardy_preset" for index in ids_a)
    assert abs(unconditional - 0.10) < 0.02
    assert abs(official / len(ids_a) - 0.10) < 0.02

    modes = Counter()
    for prompt_id in ids_a[:2000].tolist():
        mode = first.sample_control_mode(prompt_id)
        assert mode in records[prompt_id]["control_modes"]
        modes[mode] += 1
    assert set(modes) == set(CONTROL_MODES)


def test_ui_control_trajectory_shapes_and_stationary_speed_guardrail() -> None:
    records = load_prompt_bank("distill_data/text_control_v1/prompt_bank.jsonl")
    sampler = PromptControlSampler(records, torch.Generator().manual_seed(93))
    start = torch.tensor([1.0, -2.0])
    stationary_id = next(
        record["prompt_id"] for record in records if record["family"] == "idle_stance"
    )
    for mode in records[stationary_id]["control_modes"]:
        trajectory = sampler.sample_trajectory(
            stationary_id,
            mode,
            start_xz=start,
            start_heading=0.0,
            history_frames=4,
        )
        assert trajectory.root_xz.shape == (len(trajectory.frame_indices), 2)
        if len(trajectory.root_xz):
            displacement = torch.linalg.vector_norm(trajectory.root_xz - start, dim=-1)
            assert float(displacement.max()) <= 0.75
        if mode == "keyboard_heading":
            assert trajectory.heading_angles is not None
            assert len(trajectory.heading_angles) == len(trajectory.frame_indices)


def test_heading_pack_uses_teacher_normalized_features_and_mask() -> None:
    observed = torch.randn(2, 40, 330)
    mask = torch.zeros_like(observed)
    mask[0, [10, 20], 3:5] = 1
    packed = pack_heading_condition(observed, mask)
    assert packed.shape == (2, 64, 3)
    torch.testing.assert_close(packed[0, 10, :2], observed[0, 10, 3:5])
    assert packed[0, 10, 2] == 1
    assert int(torch.count_nonzero(packed[1])) == 0


if __name__ == "__main__":
    test_prompt_and_control_distribution_is_compatible_and_deterministic()
    test_ui_control_trajectory_shapes_and_stationary_speed_guardrail()
    test_heading_pack_uses_teacher_normalized_features_and_mask()
    print("CONTROL_DISTRIBUTION_PASS")
