from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
import torch
from torch import Tensor

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def compute_surprise_map(z_pred: Tensor, z_target: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute normalized L2 latent prediction error per spatial column."""
    if z_pred.shape != z_target.shape:
        raise ValueError(
            f"z_pred and z_target must have the same shape, got "
            f"{tuple(z_pred.shape)} and {tuple(z_target.shape)}"
        )
    if z_pred.ndim != 5:
        raise ValueError("Expected [B, D, T, H, W] latent tensors")
    return torch.sqrt((z_pred - z_target).pow(2).mean(dim=1) + eps)


def summarize_surprise(surprise_map: Tensor) -> dict[str, float]:
    """Return scalar logging metrics for a [B, T, H, W] surprise map."""
    if surprise_map.ndim != 4:
        raise ValueError("Expected surprise_map [B, T, H, W]")
    detached = surprise_map.detach()
    stats = {
        "surprise/global_l2": detached.mean().item(),
        "surprise/max": detached.max().item(),
        "surprise/min": detached.min().item(),
        "surprise/std": detached.std(unbiased=False).item(),
    }
    for timestep, value in enumerate(detached.mean(dim=(0, 2, 3))):
        stats[f"surprise/mean_by_timestep/{timestep}"] = value.item()
    column_means = detached.mean(dim=(0, 1))
    for row in range(column_means.shape[0]):
        for col in range(column_means.shape[1]):
            stats[f"surprise/mean_by_column/{row}_{col}"] = column_means[
                row, col
            ].item()
    return stats


def summarize_gate(gate_map: Tensor, adaptive_threshold: Tensor | float | None = None):
    if gate_map.ndim != 4:
        raise ValueError("Expected gate_map [B, T, H, W]")
    detached = gate_map.detach()
    stats = {
        "gate/mean": detached.mean().item(),
        "gate/min": detached.min().item(),
        "gate/max": detached.max().item(),
        "gate/std": detached.std(unbiased=False).item(),
    }
    if adaptive_threshold is not None:
        threshold = (
            adaptive_threshold.detach().item()
            if isinstance(adaptive_threshold, Tensor)
            else float(adaptive_threshold)
        )
        stats["gate/adaptive_threshold"] = threshold
    return stats


def _save_heatmap(path: Path, values: np.ndarray, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(values, cmap="magma")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_surprise_heatmaps(
    output_dir: str | Path,
    surprise_map: Tensor,
    step: int,
    gate_map: Tensor | None = None,
    timesteps: Iterable[int] | None = None,
    batch_index: int = 0,
) -> list[Path]:
    """Save surprise and optional gate heatmaps for selected timesteps."""
    if surprise_map.ndim != 4:
        raise ValueError("Expected surprise_map [B, T, H, W]")
    output_dir = Path(output_dir)
    n_timesteps = surprise_map.shape[1]
    if timesteps is None:
        timesteps = (0, n_timesteps - 1)
    saved_paths: list[Path] = []
    surprise_np = surprise_map.detach().float().cpu().numpy()
    gate_np = gate_map.detach().float().cpu().numpy() if gate_map is not None else None
    for timestep in timesteps:
        if timestep < 0:
            timestep = n_timesteps + timestep
        if not 0 <= timestep < n_timesteps:
            continue
        path = output_dir / f"surprise_step_{step:06d}_t{timestep}.png"
        _save_heatmap(
            path, surprise_np[batch_index, timestep], f"surprise t={timestep}"
        )
        saved_paths.append(path)
        if gate_np is not None:
            gate_path = output_dir / f"gate_step_{step:06d}_t{timestep}.png"
            _save_heatmap(
                gate_path, gate_np[batch_index, timestep], f"gate t={timestep}"
            )
            saved_paths.append(gate_path)
    return saved_paths


def save_surprise_npz(
    output_dir: str | Path,
    surprise_map: Tensor,
    step: int,
    batch_index: int,
    gate_map: Tensor | None = None,
    adaptive_threshold: Tensor | float | None = None,
) -> Path:
    """Persist sampled surprise data without storing full latent tensors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"surprise_step_{step:06d}_batch_{batch_index}.npz"
    payload = {
        "step": np.array(step),
        "batch_index": np.array(batch_index),
        "surprise_map": surprise_map.detach().float().cpu().numpy(),
    }
    if gate_map is not None:
        payload["gate_map"] = gate_map.detach().float().cpu().numpy()
    if adaptive_threshold is not None:
        threshold = (
            adaptive_threshold.detach().cpu().item()
            if isinstance(adaptive_threshold, Tensor)
            else float(adaptive_threshold)
        )
        payload["adaptive_threshold"] = np.array(threshold)
    np.savez_compressed(path, **payload)
    return path
