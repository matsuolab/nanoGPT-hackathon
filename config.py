from dataclasses import dataclass, field
from typing import ClassVar, Optional


@dataclass
class ModelConfig:
    block_size: Optional[int] = None
    vocab_size: Optional[int] = None
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None
    dropout: Optional[float] = None
    bias: Optional[bool] = None


@dataclass
class OptimizerConfig:
    name: Optional[str] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    beta1: Optional[float] = None
    beta2: Optional[float] = None


@dataclass
class TrainConfig:
    batch_size: Optional[int] = None
    local_batch_size: Optional[int] = None
    block_size: Optional[int] = None
    max_iters: Optional[int] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    grad_clip: Optional[float] = None
    decay_lr: Optional[bool] = None
    warmup_iters: Optional[int] = None
    lr_decay_iters: Optional[int] = None
    min_lr: Optional[float] = None
    eval_interval: Optional[int] = None
    log_interval: Optional[int] = None
    eval_iters: Optional[int] = None
    eval_only: Optional[bool] = None
    always_save_checkpoint: Optional[bool] = None
    init_from: Optional[str] = None


@dataclass
class DatasetConfig:
    name: Optional[str] = None
    default_config: ClassVar[Config] = field(
        default_factory=Config
    )


@dataclass
class OpenWebTextConfig(DatasetConfig):
    name: Optional[str] = None
    default_config: ClassVar[Config] = field(
        default_factory=lambda: Config(
            model = ModelConfig(
                block_size=1024,
                vocab_size=50304,
                n_layer=12,
                n_head=12,
                n_embd=768,
                dropout=0.0,
                bias=False
            ),
            train = TrainConfig(
                batch_size=480,
                local_batch_size=12,
                block_size=1024,
                max_iters=600_000,
                optimizer=field(
                    default_factory=lambda: OptimizerConfig(
                        name="adam",
                        learning_rate=6e-4,
                        weight_decay=0.1,
                        beta1=0.9,
                        beta2=0.95
                    )
                ),
                grad_clip=1.0,
                decay_lr=True,
                warmup_iters=2000,
                lr_decay_iters=600_000,
                min_lr=6e-5,
                eval_interval=2000,
                log_interval=1,
                eval_iters=200,
                eval_only=False,
                always_save_checkpoint=True,
                init_from="scratch",
            ),
            out_dir="out",
            wandb_log=False,
            wandb_project="owt",
            wandb_run_name="gpt2"
        )
    )


@dataclass
class ShakespeareConfig(DatasetConfig):
    name: str = "shakespeare"
    default_config: ClassVar[Config] = field(
        default_factory=lambda: Config(
            model = ModelConfig(
                block_size=256,
                vocab_size=65,
                n_layer=6,
                n_head=6,
                n_embd=384,
                dropout=0.2
                bias=False
            ),
            train = TrainConfig(
                batch_size=64,
                local_batch_size=64,
                block_size=256,
                max_iters=5000,
                optimizer=field(
                    default_factory=lambda: OptimizerConfig(
                        name="adam",
                        learning_rate=1e-3,
                        weight_decay=0.1,
                        beta1=0.9,
                        beta2=0.99
                    )
                ),
                grad_clip=1.0,
                decay_lr=True,
                warmup_iters=100,
                lr_decay_iters=5000,
                min_lr=1e-4,
                eval_interval=250,
                log_interval=10,
                eval_iters=200,
                eval_only=False,
                always_save_checkpoint=False,
                init_from="scratch",
            ),
            out_dir="out-shakespeare-char",
            wandb_log=False,
            wandb_project="shakespeare-char",
            wandb_run_name="mini-gpt"
        )
    )


@dataclass
class Config:
    defaults: List[Any] = field(
        default_factory=lambda: [{"dataset": "openwebtext"}, "_self_"]
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    out_dir: str = "out"
    wandb_log: bool = False
    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"

    def __post_init__(self):
        if self.train.eval_only:
            if self.train.batch_size is None:
                self.train.batch_size = 8
            if self.train.local_batch_size is None:
                self.train.local_batch_size = 8
            if self.train.eval_iters is None:
                self.train.eval_iters = 500

        for key, value in vars(self.dataset.default_config).items():
            if key == "model":
                for k_m, v_m in vars(self.dataset.default_config.model).items():
                    if getattr(self.model, k_m) is None:
                        setattr(self.model, k_m, v_m)

            elif key == "train":
                for k_t, v_t in vars(self.dataset.default_config.train).items():
                    if k_t == "optimizer":
                        for k_o, v_o in vars(self.dataset.default_config.train.optimizer).items():
                            if getattr(self.train.optimizer, k_o) is None:
                                setattr(self.train.optimizer, k_o, v_o)

                    if getattr(self.train, k_t) is None:
                        setattr(self.train, k_t, v_t)

            else:
                if getattr(self, key) is None:
                    setattr(self, key, value)
