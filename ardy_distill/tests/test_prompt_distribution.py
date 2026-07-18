from __future__ import annotations

from ardy_distill.prompt_distribution import build_prompt_bank, summarize_prompt_bank


def test_prompt_bank_is_deterministic_unique_and_control_aware() -> None:
    first = build_prompt_bank(size=512, seed=71)
    second = build_prompt_bank(size=512, seed=71)
    assert first == second
    assert first[0]["text"] == ""
    assert first[0]["prompt_id"] == 0
    assert len({record["text"] for record in first}) == len(first)
    assert [record["prompt_id"] for record in first] == list(range(len(first)))
    assert all(record["control_modes"] for record in first)
    assert all(record["speed_min_mps"] <= record["speed_max_mps"] for record in first)
    summary = summarize_prompt_bank(first, seed=71)
    assert summary["count"] == 512
    assert sum(summary["family_counts"].values()) == 512
    assert len(summary["sha256"]) == 64


if __name__ == "__main__":
    test_prompt_bank_is_deterministic_unique_and_control_aware()
    print("PROMPT_DISTRIBUTION_PASS")
