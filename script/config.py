from dataclasses import dataclass
from typing import Literal, Sequence
from argparse import ArgumentParser
from os import path
import tomllib

class PrettyPrintConfig:
    def __str__(self):
        return f"{type(self).__name__}(\n" + '\n'.join(f"    {k}={v!r}" for k, v in vars(self).items()) + "\n)"

@dataclass
class TrainingConfig(PrettyPrintConfig):
    num_epochs: int
    batch_size: int
    eval_steps: Literal["no", "epoch", "once"] | int
    save_steps: Literal["no", "epoch", "once"] | int
    gradient_accumulation_steps: int
    mixed_precision: str

@dataclass
class OptimizerConfig(PrettyPrintConfig):
    peak_lr: float
    weight_decay: float
    eps: float
    betas: tuple[float, float]
    layer_lr_decay_factor: int | None = None

@dataclass
class SchedulerConfig(PrettyPrintConfig):
    type: str
    num_warmup_steps: int | None = None
    max_steps: int | None = None

@dataclass
class UnfreezingConfig(PrettyPrintConfig):
    mode: Literal["epoch", "step"]
    schedule: list[int]

@dataclass
class Layer:
    include: list[str]
    exclude: list[str] | None = None

    def __str__(self):
        return f"{self.include}{f' - {self.exclude}' if self.exclude is not None else ''}"

@dataclass
class LayerConfig:
    layers: list[Layer]

    def __str__(self):
        return f"{type(self).__name__}(\n" + '\n'.join(f"    {layer}" for layer in self.layers) + "\n)"

@dataclass
class ScriptConfig(PrettyPrintConfig):
    model_dir: str | None
    config_file: str
    train_data: str
    eval_data: str
    continue_from_checkpoint: bool
    ignore_missing: list[str]

@dataclass
class Config:
    training_config: TrainingConfig
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    unfreezing_config: UnfreezingConfig | None
    layer_config: LayerConfig | None
    script_config: ScriptConfig

    def __str__(self):
        return '\n'.join(str(v) for v in vars(self).values() if v is not None)

    @classmethod
    def from_args(cls, args: Sequence[str] | None = None):
        parser = ArgumentParser()
        parser.add_argument("--model_dir")
        parser.add_argument("--train_data", required=True)
        parser.add_argument("--eval_data", required=True)
        parser.add_argument("--continue_from_checkpoint", action="store_true")
        parser.add_argument("--ignore_missing", nargs='+', choices=("model", "optimizer", "scheduler", "scaler", "rng_state"), default=[])
        parser.add_argument("--config_file")
        args = parser.parse_args(args)
        if args.config_file is None:
            if args.model_dir is None:
                raise ValueError("Cannot determine config file path. Please specify --model_dir or --config_file")
            args.config_file = path.join(args.model_dir, "last_config.toml")
        script_config = ScriptConfig(**vars(args))
        with open(script_config.config_file, "rb") as f:
            config = tomllib.load(f)
        config = cls(
            training_config=TrainingConfig(**config["training"]),
            optimizer_config=OptimizerConfig(**config["optimizer"]),
            scheduler_config=SchedulerConfig(**config["scheduler"]),
            unfreezing_config=UnfreezingConfig(**config["unfreezing"]) if "unfreezing" in config else None,
            layer_config=LayerConfig([Layer(**layer) for layer in config["layer"]]) if "layer" in config else None,
            script_config=script_config
        )
        if (
            config.unfreezing_config is not None or
            config.optimizer_config.layer_lr_decay_factor is not None
        ) and config.layer_config is None:
            raise ValueError("Must specify 'layer' when using 'gradual unfreezing' or 'discriminative fine-tuning'")
        if (
            config.training_config.save_steps != "no" or
            script_config.continue_from_checkpoint
        ) and config.script_config.model_dir is None:
            raise ValueError("Cannot determine model directory. Please specify --model_dir")
        if (
            config.unfreezing_config is not None and
            len(config.layer_config.layers) != len(config.unfreezing_config.schedule)
        ):
            raise ValueError(
                f"Length of 'layer' and 'schedule' must be the same. "
                f"Got {len(config.layer_config.layers)} and {len(config.unfreezing_config.schedule)}"
            )
        if (
            config.training_config.eval_steps not in ("no", "epoch", "once") and
            not isinstance(config.training_config.eval_steps, int)
        ):
            raise ValueError(
                f"'eval_steps' must be one of 'no', 'epoch', 'once' or an integer. "
                f"Got {config.training_config.eval_steps}"
            )
        if (
            config.training_config.save_steps not in ("no", "epoch", "once") and
            not isinstance(config.training_config.save_steps, int)
        ):
            raise ValueError(
                f"'save_steps' must be one of 'no', 'epoch', 'once' or an integer. "
                f"Got {config.training_config.save_steps}"
            )
        if not config.script_config.continue_from_checkpoint and config.script_config.ignore_missing:
            raise ValueError(
                f"'ignore_missing' can only be used when continuing from a checkpoint. "
                f"Got {config.script_config.ignore_missing}"
            )
        return config
