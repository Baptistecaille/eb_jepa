import torch
import torch.nn as nn

from eb_jepa.cortical_world_model import (
    CorticalActionEncoder,
    CorticalObservationEncoder,
    CorticalTemporalPredictor,
)
from eb_jepa.jepa import JEPA
from eb_jepa.losses import SquareLossSeq
from eb_jepa.state_decoder import MLPXYHead


class _ZeroRegularizer(nn.Module):
    def forward(self, x, actions=None):
        zero = x.sum() * 0.0
        return zero, zero, {}


def test_cortical_observation_encoder_maps_video_to_latent_grid():
    encoder = CorticalObservationEncoder(
        in_channels=2,
        latent_dim=16,
        grid_h=4,
        grid_w=4,
        patch_size=4,
    )

    observations = torch.randn(3, 2, 5, 16, 16)

    encoded = encoder(observations)

    assert encoded.shape == (3, 16, 5, 4, 4)


def test_cortical_observation_encoder_rejects_non_divisible_images():
    encoder = CorticalObservationEncoder(
        in_channels=2,
        latent_dim=16,
        grid_h=4,
        grid_w=4,
        patch_size=4,
    )

    observations = torch.randn(3, 2, 5, 17, 16)

    try:
        encoder(observations)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected non-divisible image size to raise ValueError")


def test_cortical_action_encoder_maps_action_sequences():
    encoder = CorticalActionEncoder(action_dim=2, embed_dim=8)

    actions = torch.randn(3, 2, 5)

    encoded = encoder(actions)

    assert encoded.shape == (3, 8, 5)


def test_cortical_temporal_predictor_one_step_shape_and_gradients():
    action_encoder = CorticalActionEncoder(action_dim=2, embed_dim=8)
    predictor = CorticalTemporalPredictor(
        latent_dim=16,
        action_embed_dim=8,
        memory_dim=32,
        grid_h=4,
        grid_w=4,
    )
    state = torch.randn(3, 16, 1, 4, 4, requires_grad=True)
    actions = torch.randn(3, 2, 1, requires_grad=True)

    encoded_actions = action_encoder(actions)
    next_state = predictor(state, encoded_actions)
    loss = next_state.square().mean()
    loss.backward()

    assert next_state.shape == (3, 16, 1, 4, 4)
    assert state.grad is not None
    assert actions.grad is not None
    assert any(p.grad is not None for p in action_encoder.parameters())
    assert any(p.grad is not None for p in predictor.parameters())


def test_cortical_components_integrate_with_jepa_autoregressive_unroll():
    encoder = CorticalObservationEncoder(
        in_channels=2,
        latent_dim=16,
        grid_h=4,
        grid_w=4,
        patch_size=4,
    )
    action_encoder = CorticalActionEncoder(action_dim=2, embed_dim=8)
    predictor = CorticalTemporalPredictor(
        latent_dim=16,
        action_embed_dim=8,
        memory_dim=32,
        grid_h=4,
        grid_w=4,
    )
    model = JEPA(
        encoder=encoder,
        aencoder=action_encoder,
        predictor=predictor,
        regularizer=_ZeroRegularizer(),
        predcost=SquareLossSeq(),
    )

    observations = torch.randn(3, 2, 5, 16, 16)
    actions = torch.randn(3, 2, 4)

    predicted_states, losses = model.unroll(
        observations,
        actions,
        nsteps=4,
        unroll_mode="autoregressive",
        ctxt_window_time=1,
        compute_loss=True,
    )

    assert predicted_states.shape == (3, 16, 5, 4, 4)
    assert len(losses) == 5
    assert losses[0].requires_grad


def test_xy_probe_accepts_spatial_cortical_latents():
    head = MLPXYHead(input_shape=16 * 4 * 4)
    states = torch.randn(3, 16, 5, 4, 4)

    predictions = head(states)

    assert predictions.shape == (3, 2, 5)
