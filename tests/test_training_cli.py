from omegaconf import OmegaConf

from eb_jepa.training_utils import parse_training_cli_args


def test_parse_training_cli_args_supports_documented_overrides():
    fname, folder, overrides = parse_training_cli_args(
        [
            "--fname",
            "examples/image_jepa/cfgs/default.yaml",
            "optim.epochs=1",
            "logging.log_wandb=false",
            "data.batch_size=8",
        ],
        default_fname="default.yaml",
    )

    assert fname == "examples/image_jepa/cfgs/default.yaml"
    assert folder is None

    cfg = OmegaConf.create(overrides)
    assert cfg.optim.epochs == 1
    assert cfg.logging.log_wandb is False
    assert cfg.data.batch_size == 8


def test_parse_training_cli_args_supports_equals_flags_and_folder():
    fname, folder, overrides = parse_training_cli_args(
        [
            "--fname=examples/video_jepa/cfgs/default.yaml",
            "--folder=checkpoints/dev",
            "--data.num_workers=0",
        ],
        default_fname="default.yaml",
    )

    assert fname == "examples/video_jepa/cfgs/default.yaml"
    assert folder == "checkpoints/dev"
    assert OmegaConf.create(overrides).data.num_workers == 0
