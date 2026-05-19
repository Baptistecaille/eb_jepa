from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CorticalObservationEncoder(nn.Module):
    """Patch-wise cortical encoder for EB-JEPA video tensors.

    Input tensors use the EB-JEPA convention [B, C, T, H, W]. Outputs keep the
    same sequence convention and expose the cortical map as spatial dimensions:
    [B, D, T, grid_h, grid_w].
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 128,
        grid_h: int = 4,
        grid_w: int = 4,
        patch_size: int = 8,
        column_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.patch_size = patch_size
        self.n_columns = grid_h * grid_w

        patch_dim = in_channels * patch_size * patch_size
        hidden_dim = column_hidden_dim or max(latent_dim * 2, patch_dim)
        self.columns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(patch_dim),
                    nn.Linear(patch_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, latent_dim),
                    nn.LayerNorm(latent_dim),
                )
                for _ in range(self.n_columns)
            ]
        )

    def _extract_patches(self, observations: Tensor) -> Tensor:
        if observations.ndim != 5:
            raise ValueError(
                "CorticalObservationEncoder expects [B, C, T, H, W] observations"
            )
        B, C, T, H, W = observations.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")
        expected_h = self.grid_h * self.patch_size
        expected_w = self.grid_w * self.patch_size
        if H != expected_h or W != expected_w:
            raise ValueError(
                "Image height and width must be divisible into the configured "
                f"cortical grid: expected {(expected_h, expected_w)}, got {(H, W)}"
            )

        frames = observations.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        ps = self.patch_size
        patches = frames.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(B, T, self.n_columns, C * ps * ps)

    def forward(self, observations: Tensor) -> Tensor:
        patches = self._extract_patches(observations)
        latents = [
            column(patches[:, :, i, :].flatten(0, 1))
            for i, column in enumerate(self.columns)
        ]
        Z = torch.stack(latents, dim=1)
        B, _, T, _ = observations.shape[:4]
        Z = Z.view(B, T, self.n_columns, self.latent_dim)
        Z = Z.view(B, T, self.grid_h, self.grid_w, self.latent_dim)
        return Z.permute(0, 4, 1, 2, 3).contiguous()


class CorticalActionEncoder(nn.Module):
    """Continuous action encoder for [B, A, T] action sequences."""

    def __init__(self, action_dim: int = 2, embed_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, actions: Tensor) -> Tensor:
        if actions.ndim != 3:
            raise ValueError("CorticalActionEncoder expects [B, A, T] actions")
        if actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected {self.action_dim} action channels")
        encoded = self.net(actions.transpose(1, 2))
        return encoded.transpose(1, 2).contiguous()


class SpatialNeighborhoodAggregator(nn.Module):
    """Aggregates four-neighbor messages over the cortical grid."""

    def __init__(self, latent_dim: int, grid_h: int, grid_w: int):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_columns = grid_h * grid_w
        self.proj = nn.Linear(latent_dim, latent_dim)

        neighbors: list[list[int]] = []
        for row in range(grid_h):
            for col in range(grid_w):
                idxs = []
                for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    n_row = row + d_row
                    n_col = col + d_col
                    if 0 <= n_row < grid_h and 0 <= n_col < grid_w:
                        idxs.append(n_row * grid_w + n_col)
                neighbors.append(idxs)
        self.neighbors = neighbors

    def forward(self, Z: Tensor) -> Tensor:
        messages = []
        for column_idx, neighbors in enumerate(self.neighbors):
            if neighbors:
                msg = Z[:, neighbors, :].mean(dim=1)
            else:
                msg = torch.zeros_like(Z[:, column_idx, :])
            messages.append(self.proj(msg))
        return torch.stack(messages, dim=1)


class ColumnStateUpdater(nn.Module):
    """Residual local update for one cortical column."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        action_embed_dim: int,
        memory_dim: int,
    ):
        super().__init__()
        in_dim = latent_dim + latent_dim + action_embed_dim + memory_dim
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.proposal = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(latent_dim + memory_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.delta = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_i: Tensor, m_i: Tensor, action: Tensor, memory: Tensor):
        fused = self.fusion(torch.cat([z_i, m_i, action, memory], dim=-1))
        proposal = self.proposal(fused)
        gate = self.gate(torch.cat([proposal, memory], dim=-1))
        return z_i + self.delta(proposal) * gate, gate


class GlobalMemory(nn.Module):
    """GRUCell memory initialized from the current pooled cortical state."""

    def __init__(self, latent_dim: int, action_embed_dim: int, memory_dim: int):
        super().__init__()
        self.init = nn.Sequential(
            nn.Linear(latent_dim, memory_dim),
            nn.Tanh(),
        )
        self.input = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
        )
        self.gru = nn.GRUCell(memory_dim, memory_dim)

    def forward(self, pooled_z: Tensor, action: Tensor) -> Tensor:
        h0 = self.init(pooled_z)
        gru_input = self.input(torch.cat([pooled_z, action], dim=-1))
        return self.gru(gru_input, h0)


class TopDownFeedback(nn.Module):
    """Projects global memory back into position-specific column feedback."""

    def __init__(self, memory_dim: int, latent_dim: int, n_columns: int):
        super().__init__()
        self.n_columns = n_columns
        self.pos_emb = nn.Embedding(n_columns, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(memory_dim + latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, memory: Tensor) -> Tensor:
        idx = torch.arange(self.n_columns, device=memory.device)
        pos = self.pos_emb(idx).unsqueeze(0).expand(memory.shape[0], -1, -1)
        memory = memory.unsqueeze(1).expand(-1, self.n_columns, -1)
        return self.net(torch.cat([memory, pos], dim=-1))


class CorticalTemporalPredictor(nn.Module):
    """Autoregressive cortical predictor compatible with JEPA.unroll."""

    is_rnn = True
    context_length = 0

    def __init__(
        self,
        latent_dim: int = 128,
        action_embed_dim: int = 64,
        memory_dim: int = 256,
        grid_h: int = 4,
        grid_w: int = 4,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_embed_dim = action_embed_dim
        self.memory_dim = memory_dim
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_columns = grid_h * grid_w
        updater_hidden_dim = hidden_dim or memory_dim

        self.neighborhood = SpatialNeighborhoodAggregator(latent_dim, grid_h, grid_w)
        self.memory = GlobalMemory(latent_dim, action_embed_dim, memory_dim)
        self.column_updater = ColumnStateUpdater(
            latent_dim=latent_dim,
            hidden_dim=updater_hidden_dim,
            action_embed_dim=action_embed_dim,
            memory_dim=memory_dim,
        )
        self.topdown = TopDownFeedback(memory_dim, latent_dim, self.n_columns)
        self.refine = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        if state.ndim != 5 or state.shape[2] != 1:
            raise ValueError(
                "CorticalTemporalPredictor expects state [B, D, 1, grid_h, grid_w]"
            )
        if action.ndim != 3 or action.shape[2] != 1:
            raise ValueError(
                "CorticalTemporalPredictor expects action [B, action_embed_dim, 1]"
            )
        B, D, _, H, W = state.shape
        if D != self.latent_dim or H != self.grid_h or W != self.grid_w:
            raise ValueError(
                f"Expected state [B, {self.latent_dim}, 1, {self.grid_h}, {self.grid_w}], "
                f"got {tuple(state.shape)}"
            )
        if action.shape[1] != self.action_embed_dim:
            raise ValueError(f"Expected {self.action_embed_dim} action channels")

        Z = state[:, :, 0].permute(0, 2, 3, 1).reshape(B, self.n_columns, D)
        action_t = action[:, :, 0]
        messages = self.neighborhood(Z)
        memory = self.memory(Z.mean(dim=1), action_t)

        local_preds = []
        for i in range(self.n_columns):
            z_next_i, _ = self.column_updater(
                Z[:, i, :],
                messages[:, i, :],
                action_t,
                memory,
            )
            local_preds.append(z_next_i)
        Z_tilde = torch.stack(local_preds, dim=1)
        feedback = self.topdown(memory)
        Z_next = Z_tilde + self.refine(torch.cat([Z_tilde, feedback], dim=-1))
        Z_next = Z_next.view(B, self.grid_h, self.grid_w, D).permute(0, 3, 1, 2)
        return Z_next.unsqueeze(2).contiguous()
