from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from examples.ac_video_jepa.main import run


def _mean_prefixed(results: dict, prefix: str) -> float | None:
    values = [value for key, value in results.items() if key.startswith(prefix)]
    if not values:
        return None
    return float(np.mean(values))


def _normalize_results(results: dict) -> dict:
    mean_rollout_surprise = _mean_prefixed(results, "val_rollout/mean_mse/")
    return {
        "success_rate": float(results.get("success_rate", np.nan)),
        "mean_final_distance": float(results.get("mean_state_dist", np.nan)),
        "mean_episode_length": float(results.get("mean_episode_length", np.nan)),
        "mean_rollout_surprise": mean_rollout_surprise,
        "mean_rollout_stability": _mean_prefixed(results, "val_rollout/std_mse/"),
        "time_per_episode": float(results.get("avg_episode_time", np.nan)),
        "raw": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cortical EB-JEPA planning and write normalized JSON."
    )
    parser.add_argument(
        "--config", required=True, help="Training/eval YAML config path"
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to load")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    cfg = OmegaConf.load(args.config)
    cfg.meta.enable_plan_eval = True
    cfg.meta.eval_only_mode = True
    cfg.meta.load_model = True
    cfg.meta.load_checkpoint = checkpoint_path.name
    cfg.eval.num_eval_episodes = args.num_episodes
    cfg.logging.log_wandb = False
    cfg.logging.tqdm_silent = True

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = run(cfg=cfg, folder=checkpoint_path.parent)
    normalized = _normalize_results(results)
    normalized["requested_num_episodes"] = args.num_episodes
    with open(output_path, "w") as f:
        json.dump(normalized, f, indent=2)


if __name__ == "__main__":
    main()
