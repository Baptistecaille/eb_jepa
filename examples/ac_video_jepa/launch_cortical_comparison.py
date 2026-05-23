from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

MODEL_OVERRIDES = {
    "impala": {
        "model.encoder_architecture": "impala",
        "model.use_surprise_gate": False,
    },
    "cortical_no_gate": {
        "model.encoder_architecture": "cortical",
        "model.use_surprise_gate": False,
    },
    "cortical_gate": {
        "model.encoder_architecture": "cortical",
        "model.use_surprise_gate": True,
    },
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    seed: int
    folder: Path
    overrides: dict[str, object]


@dataclass(frozen=True)
class JobAssignment:
    spec: ExperimentSpec
    gpu: str | None


def _format_override(key: str, value: object) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    return f"{key}={value}"


def _parse_gpus(gpus: str) -> list[str | None]:
    if gpus == "cpu":
        return [None]
    if gpus != "auto":
        return [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return [None]
    gpu_lines = [line for line in result.stdout.splitlines() if line.startswith("GPU ")]
    return [str(i) for i in range(len(gpu_lines))] or [None]


def make_experiment_specs(
    models: Iterable[str],
    seeds: Iterable[int],
    output_root: str | Path = "checkpoints/ac_video_jepa/cortical_comparison",
) -> list[ExperimentSpec]:
    output_root = Path(output_root)
    specs: list[ExperimentSpec] = []
    for model in models:
        if model not in MODEL_OVERRIDES:
            raise ValueError(
                f"Unknown model '{model}'. Choose from {sorted(MODEL_OVERRIDES)}"
            )
        for seed in seeds:
            name = f"{model}_seed{seed}"
            overrides = dict(MODEL_OVERRIDES[model])
            overrides["meta.seed"] = seed
            specs.append(
                ExperimentSpec(
                    name=name,
                    seed=seed,
                    folder=output_root / name,
                    overrides=overrides,
                )
            )
    return specs


def plan_gpu_assignments(
    specs: list[ExperimentSpec],
    gpus: list[str | None],
    jobs_per_gpu: int,
) -> list[JobAssignment]:
    if jobs_per_gpu < 1:
        raise ValueError("jobs_per_gpu must be >= 1")
    slots = [gpu for gpu in gpus for _ in range(jobs_per_gpu)]
    if not slots:
        slots = [None]
    return [
        JobAssignment(spec=spec, gpu=slots[index % len(slots)])
        for index, spec in enumerate(specs)
    ]


def _base_command(
    python_executable: str,
    config: str,
    folder: Path,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "examples.ac_video_jepa.main",
        "--fname",
        config,
        "--folder",
        str(folder),
    ]


def build_train_command(
    spec: ExperimentSpec,
    config: str,
    python_executable: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    compile_model: bool,
    save_npz_every: int,
    save_heatmaps_every: int,
) -> list[str]:
    overrides = {
        **spec.overrides,
        "meta.load_model": False,
        "meta.enable_plan_eval": False,
        "logging.log_wandb": False,
        "optim.epochs": epochs,
        "data.batch_size": batch_size,
        "data.num_workers": num_workers,
        "data.persistent_workers": False,
        "model.compile": compile_model,
        "surprise.save_npz_every": save_npz_every,
        "surprise.save_heatmaps_every": save_heatmaps_every,
    }
    return _base_command(python_executable, config, spec.folder) + [
        _format_override(key, value) for key, value in overrides.items()
    ]


def build_eval_command(
    spec: ExperimentSpec,
    config: str,
    python_executable: str,
    num_episodes: int,
    num_workers: int,
) -> list[str]:
    overrides = {
        **spec.overrides,
        "meta.eval_only_mode": True,
        "meta.enable_plan_eval": True,
        "meta.load_model": True,
        "logging.log_wandb": False,
        "eval.num_eval_episodes": num_episodes,
        "data.num_workers": num_workers,
        "data.persistent_workers": False,
    }
    return _base_command(python_executable, config, spec.folder) + [
        _format_override(key, value) for key, value in overrides.items()
    ]


def _run_command(
    assignment: JobAssignment,
    command: list[str],
    log_path: Path,
    cwd: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if assignment.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = assignment.gpu
    with open(log_path, "w") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        return process.wait()


def _run_stage(
    stage: str,
    assignments: list[JobAssignment],
    commands: dict[str, list[str]],
    max_parallel: int,
    logs_dir: Path,
    cwd: Path,
) -> None:
    print(
        f"\n[{stage}] launching {len(assignments)} jobs with max_parallel={max_parallel}"
    )
    failures = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for assignment in assignments:
            command = commands[assignment.spec.name]
            log_path = logs_dir / assignment.spec.name / f"{stage}.log"
            print(f"  start {assignment.spec.name} gpu={assignment.gpu} log={log_path}")
            future = executor.submit(
                _run_command,
                assignment,
                command,
                log_path,
                cwd,
            )
            futures[future] = assignment
        for future in as_completed(futures):
            assignment = futures[future]
            return_code = future.result()
            print(f"  done  {assignment.spec.name} rc={return_code}")
            if return_code != 0:
                failures.append((assignment.spec.name, return_code))
    if failures:
        raise RuntimeError(f"{stage} failures: {failures}")


def _print_dry_run(assignments: list[JobAssignment], commands: dict[str, list[str]]):
    for assignment in assignments:
        print(f"\n# {assignment.spec.name} gpu={assignment.gpu}")
        print(" ".join(commands[assignment.spec.name]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parallel launcher for comparable AC-video-JEPA cortical runs."
    )
    parser.add_argument(
        "--config",
        default="examples/ac_video_jepa/cfgs/train.yaml",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["impala", "cortical_no_gate", "cortical_gate"],
        choices=sorted(MODEL_OVERRIDES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 1000, 10000])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--gpus", default="auto", help="'auto', 'cpu', or '0,1,2'")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root")
    parser.add_argument("--logs-dir")
    parser.add_argument("--save-npz-every", type=int, default=1000)
    parser.add_argument("--save-heatmaps-every", type=int, default=1000)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(
        args.output_root or f"checkpoints/ac_video_jepa/cortical_comparison_{timestamp}"
    )
    logs_dir = Path(args.logs_dir or output_root / "parallel_logs")
    gpus = _parse_gpus(args.gpus)
    max_parallel = args.max_parallel or max(1, len(gpus) * args.jobs_per_gpu)
    specs = make_experiment_specs(args.models, args.seeds, output_root)
    assignments = plan_gpu_assignments(specs, gpus, args.jobs_per_gpu)
    cwd = Path.cwd()

    train_commands = {
        spec.name: build_train_command(
            spec,
            config=args.config,
            python_executable=args.python,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            compile_model=not args.no_compile,
            save_npz_every=args.save_npz_every,
            save_heatmaps_every=args.save_heatmaps_every,
        )
        for spec in specs
    }
    eval_commands = {
        spec.name: build_eval_command(
            spec,
            config=args.config,
            python_executable=args.python,
            num_episodes=args.eval_episodes,
            num_workers=args.num_workers,
        )
        for spec in specs
    }

    print(f"output_root={output_root}")
    print(f"logs_dir={logs_dir}")
    print(f"gpus={gpus}")
    print(f"max_parallel={max_parallel}")

    if args.dry_run:
        if not args.skip_train:
            print("\n## TRAIN")
            _print_dry_run(assignments, train_commands)
        if not args.skip_eval:
            print("\n## EVAL")
            _print_dry_run(assignments, eval_commands)
        return 0

    if not args.skip_train:
        _run_stage("train", assignments, train_commands, max_parallel, logs_dir, cwd)
    if not args.skip_eval:
        _run_stage("eval", assignments, eval_commands, max_parallel, logs_dir, cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
