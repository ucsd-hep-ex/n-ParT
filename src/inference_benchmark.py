import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

import os
import time
import math
import pickle
import sys
from contextlib import nullcontext
from typing import Callable, Optional, Tuple, List

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn import functional as F
from datetime import timedelta


class NormalizedLinear(nn.Linear):
    """Linear layer with normalized weights along embedding dimension"""

    def forward(self, x):
        normalized_weight = F.normalize(
            self.weight, p=2, dim=0
        )  # normalize along embedding dimension
        return F.linear(x, normalized_weight, self.bias)


class Block(nn.Module):
    def __init__(self, config, iblock):
        super().__init__()
        self.config = config

        layer = NormalizedLinear if config.use_nGPT== 1 else nn.Linear
        # Replace regular Linear layers with NormalizedLinear for attention
        self.key = layer(
            config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16
        )
        self.query = layer(
            config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16
        )
        self.value = layer(
            config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16
        )
        self.att_c_proj = layer(
            config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16
        )

        # Regular Linear layers for MLP
        self.c_fc = nn.Linear(
            config.n_embd, 2 * 4 * config.n_embd, bias=config.bias, dtype=torch.bfloat16
        )
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16
        )

        # Rest of the initialization code remains the same
        if config.use_nGPT == 0:
            self.rmsnorm_att = RMSNorm(config.n_embd)
            self.rmsnorm_mlp = RMSNorm(config.n_embd)

        if config.use_nGPT == 1:
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = torch.nn.Parameter(
                self.attn_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = torch.nn.Parameter(
                self.mlp_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.sqk_init_value = 1.0
            self.sqk_init_scaling = config.base_scale
            self.sqk = torch.nn.Parameter(
                self.sqk_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(
                self.suv_init_scaling * torch.ones(2 * 4 * config.n_embd, dtype=torch.float32)
            )

    # Rest of the Block class implementation remains the same
    def justnorm(self, x):
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, h):
        B, T, C = h.size()

        hin = h
        if self.config.use_nGPT == 0:
            hin = self.rmsnorm_att(h)

        hin = hin.to(self.query.weight.dtype)
        q = self.query(hin)
        k = self.key(hin)
        v = self.value(hin)

        q = q.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        k = k.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        v = v.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)

        # q = q.transpose(2, 1)
        # k = k.transpose(2, 1)

        if self.config.use_nGPT == 1:
            sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
                1, 1, self.config.n_head, self.config.n_embd // self.config.n_head
            )
            # print(f"sqk shape {sqk.shape}")
            # print(f"q shape {q.shape}")
            q = sqk * self.justnorm(q)
            k = sqk * self.justnorm(k)

        sqrt_head_dim = (self.config.n_embd / self.config.n_head) ** 0.5
        if self.config.use_nGPT == 0:
            softmax_scale = 1.0 / sqrt_head_dim
        if self.config.use_nGPT == 1:
            softmax_scale = sqrt_head_dim
        # y = flash_attn_func(
        #     q.to(dtype=torch.bfloat16),
        #     k.to(dtype=torch.bfloat16),
        #     v.to(dtype=torch.bfloat16),
        #     dropout_p=0.0,
        #     softmax_scale=softmax_scale,
        #     causal=True,
        #     window_size=(-1, -1),
        #     alibi_slopes=None,
        #     deterministic=True,
        # )
        
        # Regular PyTorch attention
        att = (q @ k.transpose(-2, -1)) * softmax_scale
        att = F.softmax(att, dim=-1).to(v.dtype)
        # print(f"v {v.dtype}")
        # print(f"att {att.dtype}")
        y = att @ v
        y = y.transpose(2, 1)
        # y = y.to(dtype=q.dtype)
        y = y.contiguous().view(B, T, self.config.n_embd)

        # print(f"self.att_c_proj {self.att_c_proj.weight.dtype}")
        # print(f"y {y.dtype}")
        h_att = self.att_c_proj(y)

        if self.config.use_nGPT == 0:
            h = h + h_att
        if self.config.use_nGPT == 1:
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = self.justnorm(h)  # normally, normalization is not needed
            B_norm = self.justnorm(h_att)

            # res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        hin = h
        if self.config.use_nGPT == 0:
            hin = self.rmsnorm_mlp(h)
        hin = hin.to(self.c_fc.weight.dtype)
        # print(f"self.c_fc {self.c_fc.weight.dtype}")
        # print(f"hin {hin.dtype}")
        uv = self.c_fc(hin)
        if self.config.use_nGPT == 1:
            suv = self.suv * (
                (self.suv_init_value / self.suv_init_scaling) * (self.config.n_embd**0.5)
            )
            uv = suv * uv
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        x_mlp = x_mlp.to(self.mlp_c_proj.weight.dtype)
        h_mlp = self.mlp_c_proj(x_mlp)

        if self.config.use_nGPT == 0:
            h = h + h_mlp
        if self.config.use_nGPT == 1:
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = self.justnorm(h)  # normally, normalization is not needed
            B_norm = self.justnorm(h_mlp)

            # res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        return h


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0**0.5)  # 1 / sqrt(n_embd)
    use_nGPT: int = 0
    dropout: float = 0.0
    bias: bool = False

class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight

# -----------------------------------------------------------------------------
# I/O

eval_interval = 1000
log_interval = 10
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 64 # used to simulate larger batch sizes
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
# model
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
max_iters = 600000 # total number of training iterations
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# 
time_limit_seconds = 1000000000     # stop after x seconds 
max_iters_per_launch = 1000000000   # stop after x steps of the current

use_nGPT = 1
learning_rate = 15e-4 

# model size and seqlen
if (1): 
    n_layer = 12
    n_head = 16
    n_embd = 1024
    block_size = 1024 # = context/sequence length

if (use_nGPT == 0):
    min_lr = 0.0 
    weight_decay = 0.1
    warmup_iters = 2000 
if (use_nGPT == 1):
    min_lr = 0.0
    weight_decay = 0.0
    warmup_iters = 0 

tlaunch = time.time()
print("Current Directory:", os.getcwd())
# the input configurations will overwrite all configs given above!
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

if (use_nGPT == 0):
    base_scale = 0.02 # can be interpreted as init_std
if (use_nGPT == 1):
    base_scale = 1.0 / n_embd ** 0.5


master_process = True
seed_offset = 0
ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")


out_dir='./'
if master_process:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


local_seed = seed_offset
np.random.seed(local_seed)
torch.manual_seed(local_seed)
torch.cuda.manual_seed(local_seed)


torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

device = 'cuda'
print(f"RUnning on {device}")
input_dim = 1024
x_in = torch.randn(32, 50, 1024, dtype=torch.bfloat16).to(device)
y = torch.randint(0, 1, (1024, 1), dtype=torch.bfloat16).to(device)

def run_inference_benchmark(model, x_in, num_runs=500):
    execution_times = []
    print(f"Running for {num_runs} times")
    print(f"Input shape: {x_in.shape}")

    for run in range(1, num_runs + 1):
        start_time = time.perf_counter()
        out = model(x_in)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        execution_times.append(elapsed_time)
        # print(f"Run {run}: {elapsed_time:.6f} seconds")

    average_time = np.mean(execution_times)
    std_deviation = np.std(execution_times)

    print("\nExecution Time Summary:")
    print(f"Average Time: {average_time:e} seconds")
    print(f"Standard Deviation: {std_deviation:e} seconds")

    return out

# -----------------------------------------------------------------------------
# baseline (not normalized)
model_args = dict(use_nGPT=0, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, base_scale=base_scale, 
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
# init a new model from scratch
print("Initializing a new model from scratch")
# determine the vocab size we'll use for from-scratch training
model_args['vocab_size'] = 50304
gptconf = GPTConfig(**model_args)
model_baseline = Block(gptconf, 1).to(device)
print(f"baseline model: {model_baseline}")

model_baseline(x_in)
out_baseline = run_inference_benchmark(model_baseline, x_in)

# -----------------------------------------------------------------------------
# normalized model
model_args = dict(use_nGPT=1, n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, base_scale=base_scale, 
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
# init a new model from scratch
print("Initializing a new model from scratch")
# determine the vocab size we'll use for from-scratch training
model_args['vocab_size'] = 50304
gptconf = GPTConfig(**model_args)
model = Block(gptconf, 1).to(device)
print(f"normalized model: {model}")

out = run_inference_benchmark(model, x_in)



print(f"outputs agree: {(out == out_baseline).all()}")
