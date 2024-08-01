"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

from dataclasses import asdict
import os
import time
import math
import pickle
from contextlib import nullcontext

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from config import Config, OpenWebTextConfig, ShakespeareConfig
from model import GPT

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="openwebtext", group="dataset", node=OpenWebTextConfig)
cs.store(name="shakespeare", group="dataset", node=SharespeareConfig)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    cfg = Omegaconf.to_object(cfg)

    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = True # use PyTorch 2.0 to compile the model to be faster
    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    assert cfg.train.batch_size % (ddp_world_size * cfg.train.local_batch_size) == 0
    gradient_accumulation_steps = cfg.train.batch_size // (ddp_world_size * cfg.train.local_batch_size)

    tokens_per_iter = cfg.train.batch_size * cfg.train.block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    
    if master_process:
        os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # poor man's data loader
    data_dir = os.path.join('data', cfg.dataset.name)
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - cfg.train.block_size, (cfg.train.local_batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+cfg.train.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+cfg.train.block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
    
    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    
    # model init
    if cfg.train.init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        cfg.model.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
        model = GPT(cfg.model)

    elif cfg.train.init_from == 'resume':
        print(f"Resuming training from {cfg.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(cfg.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_config = checkpoint['model_config']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            setattr(cfg.model, k, getattr(checkpoint_model_config, k))

        # create the model
        model = GPT(cfg.model)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    elif cfg.train.init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {cfg.train.init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=cfg.model.dropout)
        model = GPT.from_pretrained(cfg.train.init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            setattr(cfg.model, k, getattr(model.config, k))

    # crop down the model block size if desired, using model surgery
    if cfg.train.block_size < model.config.block_size:
        model.crop_block_size(cfg.train.block_size)
        cfg.model.block_size = cfg.train.block_size # so that the checkpoint will have the right value
    model.to(device)
    
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
   a
    # optimizer
    optimizer = model.configure_optimizers(cfg.optimizer, device_type)
    if cfg.train.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory
    
    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    
    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(cfg.train.eval_iters)
            for k in range(cfg.train.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < cfg.train.warmup_iters:
            return cfg.train.optimizer.learning_rate * it / cfg.train.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > cfg.train.lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - cfg.train.warmup_iters) / (cfg.train.lr_decay_iters - cfg.train.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return cfg.train.min_lr + coeff * (cfg.train.optimizer.learning_rate - cfg.train.min_lr)
    
    # logging
    if cfg.wandb_log and master_process:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))
    
    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    while True:
    
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if cfg.train.decay_lr else cfg.train.optimizer.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg.train.eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if cfg.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
            if losses['val'] < best_val_loss or cfg.train.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_config': cfg.model,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': cfg,
                    }
                    print(f"saving checkpoint to {cfg.out_dir}")
                    torch.save(checkpoint, os.path.join(cfg.out_dir, 'ckpt.pt'))
        if iter_num == 0 and cfg.train.eval_only:
            break
    
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if cfg.train.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
    
        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg.train.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(cfg.train.local_batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1
    
        # termination conditions
        if iter_num > cfg.train.max_iters:
            break
    
    if ddp:
        destroy_process_group()
