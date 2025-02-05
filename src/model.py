import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


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

        # Replace regular Linear layers with NormalizedLinear for attention
        self.key = NormalizedLinear(
            config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16
        )
        self.query = NormalizedLinear(
            config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16
        )
        self.value = NormalizedLinear(
            config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16
        )
        self.att_c_proj = NormalizedLinear(
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

        q = self.query(hin)
        k = self.key(hin)
        v = self.value(hin)

        q = q.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        k = k.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        v = v.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)

        q = q.transpose(2, 1)
        k = k.transpose(2, 1)

        if self.config.use_nGPT == 1:
            sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
                1, 1, self.config.n_head, self.config.n_embd // self.config.n_head
            )
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
        att = F.softmax(att, dim=-1)
        y = att @ v.transpose(2, 1)
        y = y.transpose(2, 1)
        y = y.to(dtype=q.dtype)
        y = y.contiguous().view(B, T, self.config.n_embd)

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
        uv = self.c_fc(hin)
        if self.config.use_nGPT == 1:
            suv = self.suv * (
                (self.suv_init_value / self.suv_init_scaling) * (self.config.n_embd**0.5)
            )
            uv = suv * uv
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
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
