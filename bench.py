"""
A much shorter version of train.py for benchmarking
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT
import hydra
from omegaconf import DictConfig

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'

@hydra.main(config_path="config", config_name="bench")
def main(cfg: DictConfig) -> None:
    device_type = 'cuda' if 'cuda' in cfg.device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    # data loading init
    if cfg.real_data:
        dataset = 'openwebtext'
        data_dir = os.path.join('data', dataset)
        train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        def get_batch(split):
            data = train_data # note ignore split in benchmarking script
            ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+cfg.block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+cfg.block_size]).astype(np.int64)) for i in ix])
            x, y = x.pin_memory().to(cfg.device, non_blocking=True), y.pin_memory().to(cfg.device, non_blocking=True)
            return x, y
    else:
        # alternatively, if fixed data is desired to not care about data loading
        x = torch.randint(50304, (cfg.batch_size, cfg.block_size), device=cfg.device)
        y = torch.randint(50304, (cfg.batch_size, cfg.block_size), device=cfg.device)
        get_batch = lambda split: (x, y)

    # model init
    gptconf = GPTConfig(
        block_size = cfg.block_size, # how far back does the model look? i.e. context size
        n_layer = 12, n_head = 12, n_embd = 768, # size of the model
        dropout = 0, # for determinism
        bias = cfg.bias,
    )
    model = GPT(gptconf)
    model.to(cfg.device)

    optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

    if compile:
        print("Compiling model...")
        model = torch.compile(model) # pytorch 2.0

    if cfg.profile:
        # useful docs on pytorch profiler:
        # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
        # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
        wait, warmup, active = 5, 5, 5
        num_steps = wait + warmup + active
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
            record_shapes=False,
            profile_memory=False,
            with_stack=False, # incurs an additional overhead, disable if not needed
            with_flops=True,
            with_modules=False, # only for torchscript models atm
        ) as prof:

            X, Y = get_batch('train')
            for k in range(num_steps):
                with ctx:
                    logits, loss = model(X, Y)
                X, Y = get_batch('train')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                lossf = loss.item()
                print(f"{k}/{num_steps} loss: {lossf:.4f}")

                prof.step() # notify the profiler at end of each step

    else:

        # simple benchmarking
        torch.cuda.synchronize()
        for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
            t0 = time.time()
            X, Y = get_batch('train')
            for k in range(num_steps):
                with ctx:
                    logits, loss = model(X, Y)
                X, Y = get_batch('train')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                lossf = loss.item()
                print(f"{k}/{num_steps} loss: {lossf:.4f}")
            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1-t0
            mfu = model.estimate_mfu(cfg.batch_size * 1 * num_steps, dt)
            if stage == 1:
                print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
