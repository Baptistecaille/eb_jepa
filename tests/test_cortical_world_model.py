import torch
import torch.nn as nn

from eb_jepa.cortical_world_model import (
    CorticalActionEncoder,
    CorticalObservationEncoder,
    CorticalTemporalPredictor,
    SurpriseGate,
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


def test_cortical_temporal_predictor_supports_ablation_flags():
    predictor = CorticalTemporalPredictor(
        latent_dim=16,
        action_embed_dim=8,
        memory_dim=32,
        grid_h=4,
        grid_w=4,
        use_spatial_neighborhood=False,
        use_global_memory=False,
        use_top_down_feedback=False,
    )
    state = torch.randn(3, 16, 1, 4, 4)
    actions = torch.randn(3, 8, 1)

    next_state = predictor(state, actions)

    assert next_state.shape == (3, 16, 1, 4, 4)


def test_surprise_gate_maps_low_and_high_surprise_to_configured_bounds():
    gate = SurpriseGate(
        alpha=100.0,
        initial_threshold=0.5,
        min_gate=0.05,
        max_gate=0.95,
        adaptive_threshold=False,
    )
    previous_state = torch.zeros(1, 4, 1, 2, 2)
    candidate_state = torch.ones_like(previous_state)
    surprise = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])

    corrected, gate_map = gate(previous_state, candidate_state, surprise)

    assert corrected.shape == previous_state.shape
    assert gate_map.shape == surprise.shape
    assert torch.allclose(gate_map[0, 0, :, 0], torch.full((2,), 0.05), atol=1e-3)
    assert torch.allclose(gate_map[0, 0, :, 1], torch.full((2,), 0.95), atol=1e-3)


def test_surprise_gate_updates_adaptive_threshold_without_gradient():
    gate = SurpriseGate(
        alpha=10.0,
        initial_threshold=0.1,
        threshold_momentum=0.5,
        adaptive_threshold=True,
    )
    surprise = torch.ones(2, 1, 2, 2, requires_grad=True)
    previous_state = torch.zeros(2, 4, 1, 2, 2)
    candidate_state = torch.ones_like(previous_state)

    gate(previous_state, candidate_state, surprise)

    assert gate.adaptive_threshold.requires_grad is False
    assert torch.allclose(gate.adaptive_threshold, torch.tensor(0.55))


def test_jepa_autoregressive_unroll_applies_causal_surprise_hook_only_with_loss():
    class HookedPredictor(nn.Module):
        is_rnn = True
        context_length = 0

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.hook_calls = 0

        def forward(self, state, action):
            return state + self.weight

        def apply_surprise_gate(self, previous_state, candidate_state, surprise_map):
            self.hook_calls += 1
            return previous_state, torch.zeros_like(surprise_map)

    class IdentityEncoder(nn.Module):
        def forward(self, observations):
            return observations

    predictor = HookedPredictor()
    model = JEPA(
        encoder=IdentityEncoder(),
        aencoder=nn.Identity(),
        predictor=predictor,
        regularizer=_ZeroRegularizer(),
        predcost=SquareLossSeq(),
    )
    observations = torch.zeros(2, 4, 3, 2, 2)
    actions = torch.zeros(2, 1, 2)

    predicted_states, losses = model.unroll(
        observations,
        actions,
        nsteps=2,
        unroll_mode="autoregressive",
        compute_loss=True,
    )

    assert predictor.hook_calls == 2
    assert predicted_states.shape == (2, 4, 3, 2, 2)
    assert losses[0].requires_grad

    predictor.hook_calls = 0
    model.unroll(
        observations[:, :, :1],
        actions,
        nsteps=2,
        unroll_mode="autoregressive",
        compute_loss=False,
    )
    assert predictor.hook_calls == 0


def test_xy_probe_accepts_spatial_cortical_latents():
    head = MLPXYHead(input_shape=16 * 4 * 4)
    states = torch.randn(3, 16, 5, 4, 4)

    predictions = head(states)

    assert predictions.shape == (3, 2, 5)
