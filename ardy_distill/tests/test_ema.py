import unittest

import torch
from torch import nn

from ardy_distill.ema import ModelEMA, ema_99_percent_horizon


class ModelEMATest(unittest.TestCase):
    def test_reported_horizon_replaces_99_percent_of_history(self) -> None:
        for decay in (0.9995, 0.995):
            horizon = ema_99_percent_horizon(decay)
            self.assertAlmostEqual(decay**horizon, 0.01, places=12)

    def test_update_uses_constant_decay_from_first_step(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        ema = ModelEMA(model, decay=0.75)

        with torch.no_grad():
            model.weight.fill_(2.0)
        ema.update(model)

        self.assertEqual(ema.num_updates, 1)
        self.assertTrue(torch.equal(ema.shadow["weight"], torch.tensor([[0.5]])))

    def test_resume_restores_checkpoint_decay_by_default(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        saved = ModelEMA(model, decay=0.9)
        saved.update(model)

        resumed = ModelEMA(model, decay=0.8)
        resumed.load_state_dict(saved.state_dict())

        self.assertEqual(resumed.decay, 0.9)
        self.assertEqual(resumed.num_updates, 1)

    def test_resume_can_override_only_decay(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        saved = ModelEMA(model, decay=0.9)
        with torch.no_grad():
            model.weight.fill_(3.0)
        saved.update(model)

        resumed = ModelEMA(model, decay=0.8, override_decay_on_load=True)
        resumed.load_state_dict(saved.state_dict())

        self.assertEqual(resumed.decay, 0.8)
        self.assertEqual(resumed.num_updates, saved.num_updates)
        self.assertTrue(torch.equal(resumed.shadow["weight"], saved.shadow["weight"]))


if __name__ == "__main__":
    unittest.main()
