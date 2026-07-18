from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.optim import AdamW

from ardy_distill.dmd2 import (
    discriminator_losses,
    distribution_matching_loss,
    fake_score_loss,
    generator_adversarial_loss,
)
from ardy_distill.models import (
    FlowStudentConfig,
    IndependentScoreBackboneCriticHeads,
    OneStepFlowStudent,
    ScoreBackboneCriticHead,
    clone_shared_critic_state_for_taps,
)
from ardy_distill.tools.expand_shared_critic_checkpoint import (
    expand_single_group_optimizer_state,
)
from ardy_distill.train_flow_dmd2 import (
    TrainingCounters,
    aggregate_discriminator_losses,
    aggregate_generator_adversarial_loss,
    critic_logits,
    endpoint_component_gradient_diagnostics,
    flow_teacher_features,
    guidance_update_plan,
    override_constant_learning_rate,
)
from ardy_distill.flow_matching import (
    euler_flow_denoise,
    flow_velocity_to_x0,
    make_flow_pair,
    sample_flow_time,
)


class ConstantVelocity(nn.Module):
    def __init__(self, velocity: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(
        self,
        state: torch.Tensor,
        history_hybrid: torch.Tensor,
        path_condition: torch.Tensor,
        first_heading: torch.Tensor,
        has_history: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> torch.Tensor:
        del history_hybrid, path_condition, first_heading, has_history, flow_time
        return self.velocity.expand_as(state)


class FlowTimeSamplingTest(unittest.TestCase):
    def test_legacy_joint_guidance_counter_migrates_exactly(self) -> None:
        counters = TrainingCounters()
        counters.load_state_dict(
            {
                "iterations": 701,
                "guidance_updates": 700,
                "generator_updates": 300,
            }
        )
        self.assertEqual(counters.score_updates, 700)
        self.assertEqual(counters.critic_updates, 700)
        self.assertFalse(counters.has_update_ratio_origin)

    def test_independent_update_ratio_is_anchored_at_resume_state(self) -> None:
        counters = TrainingCounters()
        counters.load_state_dict(
            {
                "iterations": 701,
                "guidance_updates": 700,
                "generator_updates": 300,
            }
        )
        plan = guidance_update_plan(
            counters,
            warmup_updates=600,
            score_updates_per_generator=2,
            critic_updates_per_generator=1,
        )
        self.assertEqual(plan, (True, True, False, 702, 701))
        self.assertEqual(counters.update_ratio_origin_generator, 300)
        self.assertEqual(counters.update_ratio_origin_score, 700)
        self.assertEqual(counters.update_ratio_origin_critic, 700)

        counters.score_updates += 1
        counters.critic_updates += 1
        plan = guidance_update_plan(
            counters,
            warmup_updates=600,
            score_updates_per_generator=2,
            critic_updates_per_generator=1,
        )
        self.assertEqual(plan, (True, False, False, 702, 701))
        counters.score_updates += 1
        counters.generator_updates += 1
        plan = guidance_update_plan(
            counters,
            warmup_updates=600,
            score_updates_per_generator=2,
            critic_updates_per_generator=1,
        )
        self.assertEqual(plan, (True, True, False, 704, 702))

    def test_independent_ratio_warmup_finishes_before_origin(self) -> None:
        counters = TrainingCounters()
        counters.score_updates = 9
        counters.critic_updates = 10
        plan = guidance_update_plan(
            counters,
            warmup_updates=10,
            score_updates_per_generator=1,
            critic_updates_per_generator=2,
        )
        self.assertEqual(plan, (True, False, True, 10, 10))
        self.assertFalse(counters.has_update_ratio_origin)

    def test_resume_lr_override_survives_constant_scheduler_step(self) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1.0e-7)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

        # Model the common failure mode: a checkpoint restored the old LR, while
        # the resumed experiment requested a different value.
        override_constant_learning_rate(optimizer, scheduler, 3.0e-8)
        self.assertEqual(optimizer.param_groups[0]["lr"], 3.0e-8)
        self.assertEqual(scheduler.base_lrs, [3.0e-8])
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 3.0e-8)

    def test_distribution_matching_surrogate_has_expected_score_gap_gradient(self) -> None:
        generated = torch.zeros(2, 10, 148, requires_grad=True)
        teacher_x0 = torch.ones_like(generated, requires_grad=True)
        fake_x0 = (teacher_x0.detach() + 0.25).requires_grad_(True)
        losses = distribution_matching_loss(
            generated,
            teacher_x0,
            fake_x0,
            torch.zeros_like(generated),
        )
        losses["dmd_total"].backward()

        # p_real - p_fake = fake_x0 - teacher_x0 = +0.25 and the
        # normalizer is mean(abs(generated - teacher_x0)) = 1.  The surrogate
        # uses 2x root weighting, with independent means over each channel set.
        expected_root = torch.full_like(generated[..., :20], 2.0 * 0.25 / (2 * 10 * 20))
        expected_body = torch.full_like(generated[..., 20:], 0.25 / (2 * 10 * 128))
        torch.testing.assert_close(generated.grad[..., :20], expected_root)
        torch.testing.assert_close(generated.grad[..., 20:], expected_body)
        self.assertIsNone(teacher_x0.grad)
        self.assertIsNone(fake_x0.grad)

    def test_fake_score_loss_uses_rectified_flow_velocity_target(self) -> None:
        predicted_velocity = torch.zeros(2, 10, 148)
        target_velocity = torch.cat(
            [
                torch.ones(2, 10, 20),
                torch.full((2, 10, 128), 2.0),
            ],
            dim=-1,
        )
        losses = fake_score_loss(predicted_velocity, target_velocity)
        self.assertEqual(float(losses["fake_score_root_mse"]), 1.0)
        self.assertEqual(float(losses["fake_score_body_mse"]), 4.0)
        self.assertEqual(float(losses["fake_score_total"]), 6.0)

    def test_all_exact_t1(self) -> None:
        values = sample_flow_time(
            128,
            torch.device("cpu"),
            exact_t1_probability=1.0,
            high_noise_probability=0.0,
        )
        torch.testing.assert_close(values, torch.ones_like(values))

    def test_uniform_remainder_is_bounded(self) -> None:
        torch.manual_seed(7)
        values = sample_flow_time(
            1024,
            torch.device("cpu"),
            exact_t1_probability=0.0,
            high_noise_probability=0.0,
        )
        self.assertTrue(bool((values >= 0.0).all()))
        self.assertTrue(bool((values <= 1.0).all()))
        self.assertGreater(float(values.std()), 0.2)

    def test_rejects_invalid_mixture(self) -> None:
        invalid = [(-0.1, 0.0), (1.1, 0.0), (0.0, -0.1), (0.0, 1.1), (0.7, 0.4)]
        for exact, high in invalid:
            with self.subTest(exact=exact, high=high), self.assertRaises(ValueError):
                sample_flow_time(
                    1,
                    torch.device("cpu"),
                    exact_t1_probability=exact,
                    high_noise_probability=high,
                )

    def test_backward_euler_direction_and_step_count(self) -> None:
        torch.manual_seed(11)
        noise = torch.randn(2, 3, 4)
        clean = torch.randn_like(noise)
        model = ConstantVelocity(noise - clean)
        condition = torch.zeros(2, 1, 1)
        has_history = torch.zeros(2, 1)
        for steps in (1, 2, 4):
            with self.subTest(steps=steps):
                result = euler_flow_denoise(
                    model,
                    noise,
                    condition,
                    condition,
                    torch.zeros(2, 2),
                    has_history,
                    steps=steps,
                )
                torch.testing.assert_close(result, clean)

    def test_velocity_reconstructs_clean_at_all_times(self) -> None:
        torch.manual_seed(13)
        clean = torch.randn(7, 10, 148)
        noise = torch.randn_like(clean)
        time = torch.tensor([[0.0], [0.05], [0.25], [0.5], [0.9], [0.999], [1.0]])
        noisy, velocity = make_flow_pair(clean, noise, time)
        reconstructed = flow_velocity_to_x0(noisy, velocity, time)
        torch.testing.assert_close(reconstructed, clean, rtol=2.0e-6, atol=2.0e-6)

    def test_frozen_teacher_features_pass_generator_input_gradients(self) -> None:
        torch.manual_seed(19)
        config = FlowStudentConfig(
            width=64,
            heads=4,
            trunk_blocks=1,
            body_blocks=1,
        )
        teacher = OneStepFlowStudent(config).eval().requires_grad_(False)
        critic = ScoreBackboneCriticHead(width=64, blocks=1)
        generated = torch.randn(2, 10, 148, requires_grad=True)
        noise = torch.randn_like(generated)
        time = torch.tensor([[1.0], [0.7]])
        history = torch.randn(2, 1, 148)
        path = torch.randn(2, 64, 3)
        heading = torch.randn(2, 2)
        has_history = torch.tensor([[0.0], [1.0]])
        reconstructed, features = flow_teacher_features(
            teacher,
            generated,
            noise,
            time,
            history,
            path,
            heading,
            has_history,
        )
        self.assertEqual(tuple(reconstructed.shape), (2, 10, 148))
        self.assertEqual(tuple(features.shape), (2, 10, 64))
        critic(features).sum().backward()
        self.assertIsNotNone(generated.grad)
        self.assertTrue(bool(torch.isfinite(generated.grad).all()))
        self.assertGreater(float(generated.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_teacher_feature_taps_preserve_prediction_shape_and_output(self) -> None:
        torch.manual_seed(21)
        config = FlowStudentConfig(
            width=64,
            heads=4,
            trunk_blocks=2,
            body_blocks=4,
        )
        teacher = OneStepFlowStudent(config).eval().requires_grad_(False)
        clean = torch.randn(2, 10, 148)
        noise = torch.randn_like(clean)
        time = torch.tensor([[0.25], [0.75]])
        history = torch.randn(2, 1, 148)
        path = torch.randn(2, 64, 3)
        heading = torch.randn(2, 2)
        has_history = torch.tensor([[0.0], [1.0]])

        predictions = []
        features = {}
        for tap in ("trunk_final", "body_pre", "body_mid", "body_final"):
            prediction, feature = flow_teacher_features(
                teacher,
                clean,
                noise,
                time,
                history,
                path,
                heading,
                has_history,
                feature_tap=tap,
            )
            predictions.append(prediction)
            features[tap] = feature
            self.assertEqual(tuple(feature.shape), (2, 10, 64))
        for prediction in predictions[1:]:
            torch.testing.assert_close(prediction, predictions[0], rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(features["body_pre"], features["body_final"]))
        self.assertFalse(torch.equal(features["trunk_final"], features["body_mid"]))

        requested = ("trunk_final", "body_mid", "body_final")
        multi_prediction, multi_features = flow_teacher_features(
            teacher,
            clean,
            noise,
            time,
            history,
            path,
            heading,
            has_history,
            feature_tap=requested,
        )
        self.assertIsInstance(multi_features, dict)
        self.assertEqual(tuple(multi_features), requested)
        torch.testing.assert_close(
            multi_prediction, predictions[0], rtol=0.0, atol=0.0
        )
        for tap in requested:
            torch.testing.assert_close(
                multi_features[tap], features[tap], rtol=0.0, atol=0.0
            )

        with self.assertRaises(ValueError):
            teacher.forward_with_features(
                clean,
                history,
                path,
                heading,
                has_history,
                time,
                feature_tap="not_a_tap",
            )
        with self.assertRaises(ValueError):
            teacher.forward_with_features(
                clean,
                history,
                path,
                heading,
                has_history,
                time,
                feature_tap=("body_final", "body_final"),
            )

    def test_shared_multitap_loss_aggregations_match_definitions(self) -> None:
        real = {
            "trunk_final": torch.tensor([0.2, -0.1]),
            "body_final": torch.tensor([0.8, 0.4]),
        }
        fake = {
            "trunk_final": torch.tensor([-0.3, 0.1]),
            "body_final": torch.tensor([0.2, -0.5]),
        }
        per_tap = {
            tap: discriminator_losses(real[tap], fake[tap]) for tap in real
        }
        mean_loss = aggregate_discriminator_losses(real, fake, "mean_loss")
        for name in ("critic_real", "critic_fake", "critic_total"):
            expected = torch.stack([value[name] for value in per_tap.values()]).mean()
            torch.testing.assert_close(mean_loss[name], expected)

        mean_logit = aggregate_discriminator_losses(real, fake, "mean_logit")
        expected_logit = discriminator_losses(
            torch.stack(list(real.values())).mean(dim=0),
            torch.stack(list(fake.values())).mean(dim=0),
        )
        for name in expected_logit:
            torch.testing.assert_close(mean_logit[name], expected_logit[name])

        generator_mean_loss, generator_per_tap = (
            aggregate_generator_adversarial_loss(fake, "mean_loss")
        )
        expected_generator = torch.stack(
            [generator_adversarial_loss(value) for value in fake.values()]
        ).mean()
        torch.testing.assert_close(generator_mean_loss, expected_generator)
        self.assertEqual(tuple(generator_per_tap), tuple(fake))
        generator_mean_logit, _ = aggregate_generator_adversarial_loss(
            fake, "mean_logit"
        )
        torch.testing.assert_close(
            generator_mean_logit,
            generator_adversarial_loss(
                torch.stack(list(fake.values())).mean(dim=0)
            ),
        )

    def test_independent_head_clone_preserves_initial_logits(self) -> None:
        torch.manual_seed(29)
        taps = ("body_mid", "body_final")
        shared = ScoreBackboneCriticHead(width=16, blocks=2)
        independent = IndependentScoreBackboneCriticHeads(
            width=16,
            taps=taps,
            blocks=2,
        )
        independent.load_state_dict(
            clone_shared_critic_state_for_taps(shared.state_dict(), taps)
        )
        features = {tap: torch.randn(3, 10, 16) for tap in taps}
        expected = {tap: shared(value) for tap, value in features.items()}
        actual = critic_logits(independent, features)
        self.assertEqual(tuple(actual), taps)
        for tap in taps:
            torch.testing.assert_close(actual[tap], expected[tap], rtol=0.0, atol=0.0)

        first_mid = next(independent.heads["body_mid"].parameters())
        first_final = next(independent.heads["body_final"].parameters())
        self.assertNotEqual(first_mid.data_ptr(), first_final.data_ptr())

    def test_independent_head_optimizer_expansion_preserves_adam_moments(self) -> None:
        torch.manual_seed(31)
        taps = ("body_mid", "body_final")
        shared = ScoreBackboneCriticHead(width=16, blocks=1)
        old_optimizer = AdamW(shared.parameters(), lr=1.0e-6, weight_decay=0)
        old_optimizer.zero_grad(set_to_none=True)
        shared(torch.randn(2, 10, 16)).sum().backward()
        old_optimizer.step()
        old_state = old_optimizer.state_dict()

        independent = IndependentScoreBackboneCriticHeads(
            width=16,
            taps=taps,
            blocks=1,
        )
        independent.load_state_dict(
            clone_shared_critic_state_for_taps(shared.state_dict(), taps)
        )
        new_optimizer = AdamW(
            independent.parameters(), lr=1.0e-6, weight_decay=0
        )
        expanded = expand_single_group_optimizer_state(old_state, copies=2)
        new_optimizer.load_state_dict(expanded)

        old_parameters = list(shared.parameters())
        new_parameters = list(independent.parameters())
        self.assertEqual(len(new_parameters), 2 * len(old_parameters))
        for copy_index in range(2):
            for index, old_parameter in enumerate(old_parameters):
                new_parameter = new_parameters[copy_index * len(old_parameters) + index]
                self.assertEqual(new_parameter.shape, old_parameter.shape)
                old_moments = old_optimizer.state[old_parameter]
                new_moments = new_optimizer.state[new_parameter]
                torch.testing.assert_close(
                    new_moments["exp_avg"], old_moments["exp_avg"]
                )
                torch.testing.assert_close(
                    new_moments["exp_avg_sq"], old_moments["exp_avg_sq"]
                )
                self.assertEqual(
                    int(new_moments["step"].item()),
                    int(old_moments["step"].item()),
                )

    def test_endpoint_component_gradient_diagnostics_match_exact_sum(self) -> None:
        generated = torch.tensor([1.0, 2.0], requires_grad=True)
        dmd = generated[0]
        adversarial = 2.0 * generated[1]
        regularizer = -generated[0] + generated[1]
        total = 2.0 * dmd + 3.0 * adversarial + 0.5 * regularizer

        metrics = endpoint_component_gradient_diagnostics(
            generated,
            total,
            {
                "dmd": (dmd, 2.0),
                "adversarial": (adversarial, 3.0),
                "regularizer": (regularizer, 0.5),
            },
        )
        expected_total = torch.tensor([1.5, 6.5])
        expected_rms = expected_total.square().mean().sqrt()
        torch.testing.assert_close(
            metrics["generator_component_total_endpoint_grad_rms"], expected_rms
        )
        torch.testing.assert_close(
            metrics["generator_component_sum_endpoint_grad_rms"], expected_rms
        )
        torch.testing.assert_close(
            metrics["generator_component_sum_vs_total_endpoint_grad_max_abs"],
            torch.tensor(0.0),
        )
        torch.testing.assert_close(
            metrics["generator_component_sum_vs_total_endpoint_grad_relative_rms"],
            torch.tensor(0.0),
        )
        torch.testing.assert_close(
            metrics["generator_component_dmd_vs_adversarial_endpoint_cosine"],
            torch.tensor(0.0),
        )

    def test_exact_t1_paired_noise_erases_real_fake_critic_signal(self) -> None:
        torch.manual_seed(23)
        config = FlowStudentConfig(
            width=64,
            heads=4,
            trunk_blocks=1,
            body_blocks=1,
        )
        teacher = OneStepFlowStudent(config).eval().requires_grad_(False)
        real = torch.randn(3, 10, 148)
        fake = torch.randn_like(real)
        shared_noise = torch.randn_like(real)
        exact_t1 = torch.ones(3, 1)
        history = torch.randn(3, 1, 148)
        path = torch.randn(3, 64, 3)
        heading = torch.randn(3, 2)
        has_history = torch.tensor([[0.0], [1.0], [1.0]])

        _, real_features = flow_teacher_features(
            teacher,
            real,
            shared_noise,
            exact_t1,
            history,
            path,
            heading,
            has_history,
        )
        _, fake_features = flow_teacher_features(
            teacher,
            fake,
            shared_noise,
            exact_t1,
            history,
            path,
            heading,
            has_history,
        )
        torch.testing.assert_close(real_features, fake_features, rtol=0.0, atol=0.0)

    def test_rejects_nonpositive_solver_steps(self) -> None:
        tensor = torch.zeros(1, 1, 1)
        with self.assertRaises(ValueError):
            euler_flow_denoise(
                ConstantVelocity(tensor),
                tensor,
                tensor,
                tensor,
                torch.zeros(1, 2),
                torch.zeros(1, 1),
                steps=0,
            )


if __name__ == "__main__":
    unittest.main()
