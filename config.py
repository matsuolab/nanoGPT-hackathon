from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


@dataclass
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95


@dataclass
class TrainConfig:
    batch_size: int = 480
    local_batch_size: int = 12
    block_size: int = 1024
    max_iters: int = 600_000
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600_000
    min_lr = 6e-5
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"


@dataclass
class DatasetConfig:
    name: Optional[str] = None


@dataclass
class OpenWebTextConfig:
    name: str = "openwebtext"


@dataclass
class OpenWebTextConfig(DatasetConfig):
    name: str = "openwebtext"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    out_dir: str = "out"
    wandb_log: bool = False
    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"

