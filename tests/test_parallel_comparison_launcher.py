from pathlib import Path

from examples.ac_video_jepa.launch_cortical_comparison import (
    ExperimentSpec,
    build_eval_command,
    build_train_command,
    make_experiment_specs,
    plan_gpu_assignments,
)


def test_make_experiment_specs_expands_models_and_seeds():
    specs = make_experiment_specs(models=["impala", "cortical_gate"], seeds=[1, 2])

    assert [spec.name for spec in specs] == [
        "impala_seed1",
        "impala_seed2",
        "cortical_gate_seed1",
        "cortical_gate_seed2",
    ]
    assert specs[0].overrides["model.encoder_architecture"] == "impala"
    assert specs[2].overrides["model.encoder_architecture"] == "cortical"
    assert specs[2].overrides["model.use_surprise_gate"] is True


def test_plan_gpu_assignments_uses_round_robin_slots():
    specs = make_experiment_specs(models=["impala", "cortical_no_gate"], seeds=[1, 2])

    assignments = plan_gpu_assignments(specs, gpus=["0", "1"], jobs_per_gpu=1)

    assert [assignment.gpu for assignment in assignments] == ["0", "1", "0", "1"]


def test_build_train_command_includes_fast_comparable_overrides(tmp_path: Path):
    spec = ExperimentSpec(
        name="cortical_gate_seed1",
        seed=1,
        folder=tmp_path / "cortical_gate_seed1",
        overrides={
            "model.encoder_architecture": "cortical",
            "model.use_surprise_gate": True,
        },
    )

    command = build_train_command(
        spec,
        config="examples/ac_video_jepa/cfgs/train.yaml",
        python_executable="python",
        epochs=12,
        batch_size=128,
        num_workers=4,
        compile_model=True,
        save_npz_every=1000,
        save_heatmaps_every=1000,
    )

    joined = " ".join(command)
    assert "python -m examples.ac_video_jepa.main" in joined
    assert f"--folder {spec.folder}" in joined
    assert "model.encoder_architecture=cortical" in command
    assert "model.use_surprise_gate=true" in command
    assert "meta.enable_plan_eval=false" in command
    assert "data.num_workers=4" in command


def test_build_eval_command_loads_trained_folder(tmp_path: Path):
    spec = ExperimentSpec(
        name="impala_seed1",
        seed=1,
        folder=tmp_path / "impala_seed1",
        overrides={"model.encoder_architecture": "impala"},
    )

    command = build_eval_command(
        spec,
        config="examples/ac_video_jepa/cfgs/train.yaml",
        python_executable="python",
        num_episodes=100,
        num_workers=2,
    )

    assert "--folder" in command
    assert str(spec.folder) in command
    assert "meta.eval_only_mode=true" in command
    assert "meta.load_model=true" in command
    assert "eval.num_eval_episodes=100" in command
