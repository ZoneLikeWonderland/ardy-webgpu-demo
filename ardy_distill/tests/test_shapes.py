from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from ardy_distill.dmd2 import (
    distribution_matching_loss,
    epsilon_to_x0,
    fake_score_loss,
    q_sample,
    q_sample_with_exact_clean,
    sample_dmd_timesteps,
)
from ardy_distill.models import (
    CodecStudentConfig,
    FlowStudentConfig,
    HistoryEncoderStudent,
    MotionDecoderStudent,
    OneStepFlowStudent,
    ScoreBackboneCriticHead,
    TemporalMotionCritic,
    build_root_projection_basis,
    project_root_trajectory,
)
from ardy_distill.runtime import save_safetensor_weights
from ardy_distill.train_codec import load_expandable_codec_weights
from ardy_distill.train_dmd2 import generate_endpoint
from ardy_distill.train_flow import load_expandable_flow_weights


def test_expandable_codec_initialization_is_function_preserving() -> None:
    torch.manual_seed(17)
    shallow = CodecStudentConfig(encoder_blocks=3, decoder_blocks=4)
    expanded = CodecStudentConfig(encoder_blocks=4, decoder_blocks=8)
    source_encoder = HistoryEncoderStudent(shallow).eval()
    source_decoder = MotionDecoderStudent(shallow).eval()
    target_encoder = HistoryEncoderStudent(expanded).eval()
    target_decoder = MotionDecoderStudent(expanded).eval()

    body = torch.randn(2, 4, 325)
    latent = torch.randn(2, 11, 128)
    root = torch.randn(2, 44, 4)
    valid = torch.ones(2, 11)
    valid[1, 0] = 0
    with torch.no_grad():
        encoder_reference = source_encoder(body)
        decoder_reference = source_decoder(latent, root, valid)

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        encoder_path = directory / "encoder.safetensors"
        decoder_path = directory / "decoder.safetensors"
        save_safetensor_weights(source_encoder, encoder_path)
        save_safetensor_weights(source_decoder, decoder_path)
        encoder_missing = load_expandable_codec_weights(target_encoder, encoder_path)
        decoder_missing = load_expandable_codec_weights(target_decoder, decoder_path)

    assert encoder_missing and decoder_missing
    assert all(name.startswith("blocks.3.") for name in encoder_missing)
    assert all(
        any(name.startswith(f"blocks.{index}.") for index in range(4, 8))
        for name in decoder_missing
    )
    with torch.no_grad():
        torch.testing.assert_close(target_encoder(body), encoder_reference, rtol=0, atol=0)
        torch.testing.assert_close(
            target_decoder(latent, root, valid), decoder_reference, rtol=0, atol=0
        )


def test_expandable_flow_initialization_is_function_preserving() -> None:
    torch.manual_seed(23)
    shallow_config = FlowStudentConfig(
        width=128,
        heads=4,
        trunk_blocks=5,
        body_blocks=2,
    )
    expanded_config = FlowStudentConfig(
        width=128,
        heads=4,
        trunk_blocks=8,
        body_blocks=8,
    )
    source = OneStepFlowStudent(shallow_config).eval()
    target = OneStepFlowStudent(expanded_config).eval()
    inputs = (
        torch.randn(2, 10, 148),
        torch.randn(2, 1, 148),
        torch.randn(2, 64, 3),
        torch.randn(2, 2),
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([[1.0], [0.75]]),
    )
    with torch.no_grad():
        reference = source(*inputs)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "flow.safetensors"
        save_safetensor_weights(source, path)
        missing = load_expandable_flow_weights(target, path)
    assert missing
    assert all(
        name.startswith("trunk.") or name.startswith("body_refiner.")
        for name in missing
    )
    with torch.no_grad():
        torch.testing.assert_close(target(*inputs), reference, rtol=0, atol=0)


def test_text_condition_warm_start_is_function_preserving() -> None:
    """Adding Qwen conditioning must not erase the converged path-only EMA."""

    torch.manual_seed(29)
    base_config = FlowStudentConfig(
        width=64,
        heads=4,
        trunk_blocks=2,
        body_blocks=2,
    )
    text_config = FlowStudentConfig(
        width=64,
        heads=4,
        trunk_blocks=2,
        body_blocks=2,
        text_feature_dim=96,
        heading_condition_features=3,
    )
    source = OneStepFlowStudent(base_config).eval()
    target = OneStepFlowStudent(text_config).eval()
    incompatible = target.load_state_dict(source.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == {
        "text_proj.weight",
        "heading_proj.weight",
        "heading_proj.bias",
    }

    inputs = (
        torch.randn(2, 10, 148),
        torch.randn(2, 1, 148),
        torch.randn(2, 64, 3),
        torch.randn(2, 2),
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([[1.0], [0.75]]),
    )
    text_feature = torch.randn(2, text_config.text_feature_dim)
    heading_condition = torch.randn(2, 64, text_config.heading_condition_features)
    with torch.no_grad():
        reference = source(*inputs)
        warm_started = target(*inputs, text_feature, heading_condition)
    torch.testing.assert_close(warm_started, reference, rtol=0, atol=0)

    assert target.text_proj is not None
    with torch.no_grad():
        target.text_proj.weight.normal_(std=0.01)
        unconditional = target(*inputs, torch.zeros_like(text_feature))
        conditioned = target(*inputs, text_feature)
        conditioned_rescaled = target(*inputs, text_feature * 128.0)
    assert not torch.equal(conditioned, unconditional)
    torch.testing.assert_close(conditioned_rescaled, conditioned, rtol=2.0e-5, atol=2.0e-6)
    assert target.heading_proj is not None
    with torch.no_grad():
        target.text_proj.weight.zero_()
        target.heading_proj.weight.normal_(std=0.01)
        no_heading = target(
            *inputs,
            torch.zeros_like(text_feature),
            torch.zeros_like(heading_condition),
        )
        with_heading = target(
            *inputs,
            torch.zeros_like(text_feature),
            heading_condition,
        )
    assert not torch.equal(with_heading, no_heading)
    assert sum(parameter.numel() for parameter in target.parameters()) - sum(
        parameter.numel() for parameter in source.parameters()
    ) == (
        text_config.text_feature_dim * text_config.width
        + text_config.path_frames_per_token
        * text_config.heading_condition_features
        * text_config.width
        + text_config.width
    )


def test_root_projection_bases() -> None:
    clean = torch.randn(2, 10, 148)
    root = clean[..., :20].reshape(2, 40, 5)
    legacy = root
    for _ in range(4):
        padded = torch.cat(
            [legacy[:, :1], legacy[:, :1], legacy, legacy[:, -1:], legacy[:, -1:]],
            dim=1,
        )
        filtered = (
            padded[:, 0:40]
            + 4.0 * padded[:, 1:41]
            + 6.0 * padded[:, 2:42]
            + 4.0 * padded[:, 3:43]
            + padded[:, 4:44]
        ) / 16.0
        legacy = torch.cat([legacy[:, :1], filtered[:, 1:-1], legacy[:, -1:]], dim=1)
    dense = project_root_trajectory(clean, 4)
    assert torch.allclose(dense[..., :20].reshape(2, 40, 5), legacy, atol=2.0e-7)
    assert torch.equal(dense[..., 20:], clean[..., 20:])

    cubic_basis = build_root_projection_basis(
        kind="cubic_controls", passes=0, control_points=10
    )
    assert int(torch.linalg.matrix_rank(cubic_basis)) == 10
    positions = torch.linspace(0, 39, 10).round().long()
    assert torch.equal(cubic_basis[positions][:, positions], torch.eye(10))
    frame = torch.arange(40, dtype=torch.float32)
    linear_root = torch.stack(
        [0.1 * frame + offset for offset in torch.arange(5, dtype=torch.float32)],
        dim=-1,
    ).unsqueeze(0)
    linear_clean = torch.zeros(1, 10, 148)
    linear_clean[..., :20] = linear_root.reshape(1, 10, 20)
    reconstructed = project_root_trajectory(
        linear_clean,
        0,
        kind="cubic_controls",
        control_points=10,
        basis=cubic_basis,
    )
    assert torch.allclose(
        reconstructed[..., :20].reshape(1, 40, 5), linear_root, atol=5.0e-7
    )


def test_student_shapes_and_finiteness() -> None:
    encoder = HistoryEncoderStudent().eval()
    flow = OneStepFlowStudent().eval()
    decoder = MotionDecoderStudent().eval()

    with torch.inference_mode():
        history_latent = encoder(torch.randn(2, 4, 325))
        generated = flow.denoise_once(
            torch.randn(2, 10, 148),
            torch.randn(2, 1, 148),
            torch.randn(2, 64, 3),
            torch.randn(2, 2),
            torch.ones(2, 1),
        )
        generated_nfe4 = flow.denoise_steps(
            torch.randn(2, 10, 148),
            torch.randn(2, 1, 148),
            torch.randn(2, 64, 3),
            torch.randn(2, 2),
            torch.ones(2, 1),
            steps=4,
        )
        body = decoder(
            torch.randn(2, 11, 128),
            torch.randn(2, 44, 4),
            torch.ones(2, 11),
        )

    assert history_latent.shape == (2, 1, 128)
    assert generated.shape == (2, 10, 148)
    assert generated_nfe4.shape == (2, 10, 148)
    assert body.shape == (2, 44, 325)
    assert torch.isfinite(history_latent).all()
    assert torch.isfinite(generated).all()
    assert torch.isfinite(generated_nfe4).all()
    assert torch.isfinite(body).all()

    probe_noise = torch.randn(2, 10, 148)
    probe_history = torch.randn(2, 1, 148)
    probe_path = torch.randn(2, 64, 3)
    probe_heading = torch.randn(2, 2)
    probe_has_history = torch.ones(2, 1)
    probe_time = torch.ones(2, 1)
    with torch.inference_mode():
        public_prediction = flow(
            probe_noise,
            probe_history,
            probe_path,
            probe_heading,
            probe_has_history,
            probe_time,
        )
        feature_prediction, bottleneck = flow.forward_with_features(
            probe_noise,
            probe_history,
            probe_path,
            probe_heading,
            probe_has_history,
            probe_time,
        )
    assert torch.equal(public_prediction, feature_prediction)
    assert bottleneck.shape == (2, 10, flow.config.width)

    smooth_flow = OneStepFlowStudent(
        FlowStudentConfig(root_smoothing_passes=4)
    ).eval()
    smooth_noise = torch.randn(2, 10, 148)
    smooth_history = torch.randn(2, 1, 148)
    smooth_path = torch.randn(2, 64, 3)
    smooth_heading = torch.randn(2, 2)
    smooth_has_history = torch.ones(2, 1)
    with torch.inference_mode():
        smooth_generated = smooth_flow.denoise_once(
            smooth_noise,
            smooth_history,
            smooth_path,
            smooth_heading,
            smooth_has_history,
        )
        training_generated = generate_endpoint(
            smooth_flow,
            {
                "initial_noise": smooth_noise,
                "path_condition": smooth_path,
                "first_heading": smooth_heading,
                "has_history": smooth_has_history,
            },
            smooth_history,
        )
    assert smooth_generated.shape == (2, 10, 148)
    assert torch.isfinite(smooth_generated).all()
    assert torch.equal(smooth_generated, training_generated)


def test_dmd2_and_critic_shapes() -> None:
    critic = TemporalMotionCritic().eval()
    history = torch.randn(2, 1, 148)
    path = torch.randn(2, 64, 3)
    path[..., 2] = (torch.rand(2, 64) > 0.8).float()
    heading = torch.randn(2, 2)
    has_history = torch.tensor([[0.0], [1.0]])
    logits = critic(
        torch.randn(2, 40, 658),
        history,
        path,
        heading,
        has_history,
        torch.tensor([0, 9]),
    )
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()

    score = OneStepFlowStudent(FlowStudentConfig(width=384, heads=6)).eval()
    score_head = ScoreBackboneCriticHead(width=384).eval()
    score_input = torch.randn(2, 10, 148, requires_grad=True)
    score_prediction, score_features = score.forward_with_features(
        score_input,
        history,
        path,
        heading,
        has_history,
        torch.tensor([[0.2], [0.8]]),
    )
    score_logits = score_head(score_features)
    assert score_prediction.shape == (2, 10, 148)
    assert score_features.shape == (2, 10, 384)
    assert score_logits.shape == (2,)
    assert torch.isfinite(score_logits).all()
    score_logits.sum().backward()
    assert score_input.grad is not None
    assert torch.isfinite(score_input.grad).all()

    generated = torch.randn(2, 10, 148, requires_grad=True)
    noise = torch.randn_like(generated)
    timesteps = torch.tensor([2, 7])
    alphas = torch.linspace(0.98, 0.001, 10)
    xt = q_sample(generated.detach(), timesteps, noise, alphas)
    mixed_xt = q_sample_with_exact_clean(
        generated.detach(),
        timesteps,
        noise,
        alphas,
        torch.tensor([True, False]),
    )
    torch.testing.assert_close(mixed_xt[0], generated[0].float())
    torch.testing.assert_close(mixed_xt[1], xt[1])
    fake_x0 = epsilon_to_x0(xt, torch.randn_like(xt), timesteps, alphas)
    score_losses = fake_score_loss(torch.randn_like(noise), noise)
    dmd_losses = distribution_matching_loss(
        generated,
        torch.randn_like(generated),
        fake_x0,
        xt,
    )
    (score_losses["fake_score_total"] + dmd_losses["dmd_total"]).backward()
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()

    # DMD2 normalizes by the generated-to-teacher residual, not by the
    # arbitrarily noised score-query input.
    probe_generated = torch.zeros(1, 1, 148, requires_grad=True)
    probe_teacher = torch.ones_like(probe_generated)
    probe_fake = torch.full_like(probe_generated, 3.0)
    probe_xt = torch.full_like(probe_generated, 100.0)
    probe = distribution_matching_loss(
        probe_generated,
        probe_teacher,
        probe_fake,
        probe_xt,
    )
    assert torch.allclose(probe["dmd_normalizer"], torch.tensor(1.0))
    assert torch.allclose(probe["dmd_gradient_abs"], torch.tensor(2.0))

    sampled = sample_dmd_timesteps(
        20_000,
        torch.device("cpu"),
        minimum=1,
        maximum=8,
        exact_max_probability=0.50,
        high_noise_probability=0.35,
    )
    assert int(sampled.min()) >= 1
    assert int(sampled.max()) <= 8
    assert float((sampled == 8).float().mean()) > 0.55

    uniform_sampled = sample_dmd_timesteps(
        90_000,
        torch.device("cpu"),
        minimum=0,
        maximum=8,
        exact_max_probability=0.0,
        high_noise_probability=0.0,
    )
    histogram = torch.bincount(uniform_sampled, minlength=9).float()
    histogram = histogram / histogram.sum()
    assert torch.all((histogram - 1.0 / 9.0).abs() < 0.01)


if __name__ == "__main__":
    test_expandable_codec_initialization_is_function_preserving()
    test_expandable_flow_initialization_is_function_preserving()
    test_text_condition_warm_start_is_function_preserving()
    test_root_projection_bases()
    test_student_shapes_and_finiteness()
    test_dmd2_and_critic_shapes()
    print("STUDENT_SHAPES_PASS")
