from __future__ import annotations

import torch

from ardy_distill.text_features import pool_qwen_hidden_states


def test_qwen_pooling_concatenates_full_layers_and_ignores_padding() -> None:
    hidden = [torch.zeros(2, 4, 2560) for _ in range(28)]
    hidden[9][:] = 1
    hidden[18][:] = 2
    hidden[27][:] = 3
    hidden[9][0, 2:] = 100
    hidden[18][0, 2:] = 200
    hidden[27][0, 2:] = 300
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    pooled = pool_qwen_hidden_states(hidden, mask)
    assert pooled.shape == (2, 7680)
    torch.testing.assert_close(pooled[:, :2560], torch.ones(2, 2560))
    torch.testing.assert_close(pooled[:, 2560:5120], torch.full((2, 2560), 2.0))
    torch.testing.assert_close(pooled[:, 5120:], torch.full((2, 2560), 3.0))


if __name__ == "__main__":
    test_qwen_pooling_concatenates_full_layers_and_ignores_padding()
    print("TEXT_FEATURES_PASS")
