from pathlib import Path

import numpy as np
import torch

from eb_jepa.surprise import (
    compute_surprise_map,
    save_surprise_npz,
    save_surprise_heatmaps,
    summarize_surprise,
)


def test_compute_surprise_map_returns_normalized_l2_per_column():
    z_target = torch.zeros(2, 4, 3, 2, 2)
    z_pred = torch.ones_like(z_target) * 2.0

    surprise = compute_surprise_map(z_pred, z_target, eps=0.0)

    assert surprise.shape == (2, 3, 2, 2)
    assert torch.allclose(surprise, torch.full((2, 3, 2, 2), 2.0))


def test_compute_surprise_map_is_zero_for_identical_latents():
    z = torch.randn(2, 4, 3, 2, 2)

    surprise = compute_surprise_map(z, z, eps=0.0)

    assert torch.count_nonzero(surprise) == 0


def test_summarize_surprise_exposes_scalar_and_vector_stats():
    surprise = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ]
        ]
    )

    stats = summarize_surprise(surprise)

    assert stats["surprise/global_l2"] == torch.mean(surprise).item()
    assert stats["surprise/max"] == 8.0
    assert stats["surprise/min"] == 1.0
    assert "surprise/mean_by_timestep/0" in stats
    assert "surprise/mean_by_column/1_1" in stats


def test_save_surprise_artifacts_create_local_files(tmp_path: Path):
    surprise = torch.arange(8, dtype=torch.float32).view(1, 2, 2, 2)
    gate = torch.ones_like(surprise) * 0.5

    heatmap_paths = save_surprise_heatmaps(
        tmp_path / "heatmaps",
        surprise,
        step=5,
        gate_map=gate,
        timesteps=(0, 1),
    )
    npz_path = save_surprise_npz(
        tmp_path / "npz",
        surprise,
        step=5,
        batch_index=0,
        gate_map=gate,
        adaptive_threshold=0.25,
    )

    assert len(heatmap_paths) == 4
    assert all(path.exists() for path in heatmap_paths)
    assert npz_path.exists()
    data = np.load(npz_path)
    assert data["surprise_map"].shape == (1, 2, 2, 2)
    assert data["gate_map"].shape == (1, 2, 2, 2)
    assert data["adaptive_threshold"] == 0.25
