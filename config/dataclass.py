#dataclass
from dataclasses import dataclass

@dataclass
class Config:
    # General settings
    out_dir: str = 'out'
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False # if True, script exits right after the first eval
    always_save_checkpoint: bool = True # if True, always save a checkpoint after each eval
    init_from: str = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

    # wandb logging
    wandb_log: bool = False # disabled by default
    wandb_project: str = 'owt'
    wandb_run_name: str = 'gpt2' # 'run' + str(time.time())

    # Data settings
    dataset: str = 'openwebtext'
    gradient_accumulation_steps: int = 40 # used to simulate larger batch sizes (5 * 8)
    batch_size: int = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 1024

    # Model settings
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False # do we use bias inside LayerNorm and Linear layers?

    # AdamW optimizer settings
    learning_rate: float = 0.0006 # max learning rate (6e-4)
    max_iters: int = 600000 # total number of training iterations
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0 # clip gradients at this value, or disable if == 0.0

    # Learning rate decay settings
    decay_lr: bool = True # whether to decay the learning rate
    warmup_iters: int = 2000 # how many steps to warm up for
    lr_decay_iters: int = 600000 # should be ~= max_iters per Chinchilla
    min_lr: float = 0.00006 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # DDP settings
    ddp: str = 'torch'
    backend: str = 'nccl' # 'nccl', 'gloo', etc.
    start: str = "\n"  # or "" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples: int = 10  # number of samples to draw
    max_new_tokens: int = 500  # number of tokens generated in each sample
    temperature: float = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
    bias: bool = False
    real_data: bool = True
    seed: int = 1337
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster
    profile: bool = False  # use pytorch profiler, or just simple bench
    device: str = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks