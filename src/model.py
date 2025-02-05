import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class Block(nn.Module):
    def __init__(self, config, iblock):
        super().__init__()
        self.config = config

        # Combined QKV projection
        self.qkv = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias, dtype=torch.bfloat16
        )
        self.att_c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias, dtype=torch.bfloat16
        )

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
    def justnorm(self, x, eps=1e-6):
        norm = x.norm(p=2, dim=-1, keepdim=True)
        # Check for NaN values in norm
        if torch.isnan(norm).any():
            raise ValueError("NaN values detected in norm calculation")
        res = x / (norm + eps)
        # Check for NaN values in result
        if torch.isnan(res).any():
            raise ValueError("NaN values detected in normalization result")
        return res

    def forward(self, h):
        B, T, C = h.size()
        if C != self.config.n_embd:
            raise ValueError(f"Expected embedding dim {self.config.n_embd}, got {C}")

        # Cast input to desired dtype at the start
        h = h.to(dtype=torch.bfloat16)

        hin = h
        if self.config.use_nGPT == 0:
            hin = self.rmsnorm_att(h)

        # Split QKV
        qkv = self.qkv(hin)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        k = k.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        v = v.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)

        # q = q.transpose(2, 1)
        # k = k.transpose(2, 1)

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

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Use PyTorch 2.0's native flash attention when available
            y = F.scaled_dot_product_attention(
                q.to(dtype=torch.bfloat16),
                k.to(dtype=torch.bfloat16),
                v.to(dtype=torch.bfloat16),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=softmax_scale,
            )

        y = y.transpose(2, 1)
        y = y.contiguous().view(B, T, self.config.n_embd)

        y = y.to(dtype=torch.bfloat16)
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

        hin = h.to(dtype=torch.bfloat16)
        if self.config.use_nGPT == 0:
            hin = self.rmsnorm_mlp(h)
        uv = self.c_fc(hin)
        if self.config.use_nGPT == 1:
            suv = self.suv * (
                (self.suv_init_value / self.suv_init_scaling) * (self.config.n_embd**0.5)
            )
            uv = suv * uv
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = (u * self.silu(v)).to(dtype=torch.bfloat16)
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

        return h.to(dtype=torch.bfloat16)


@dataclass
class GPTConfig:
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0**0.5)  # 1 / sqrt(n_embd)
    use_nGPT: int = 0
    dropout: float = 0.0
    bias: bool = True
    input_dim: int = 4
    output_dim: int = 1024

    def __post_init__(self):
        # Validate and adjust parameters
        if self.n_embd > 0:
            self.base_scale = min(1.0 / (self.n_embd**0.5), 0.1)  # Cap the scale
        else:
            raise ValueError("n_embd must be positive")

        if self.n_head > self.n_embd:
            raise ValueError("n_head cannot be larger than n_embd")


class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        # Check for NaN values in norm
        if torch.isnan(norm).any():
            raise ValueError("NaN values detected in RMSNorm calculation")
        xnorm = x * torch.rsqrt(norm + self.eps)
        if torch.isnan(xnorm).any():
            raise ValueError("NaN values detected in RMSNorm result")
        # xnorm = xnorm.to(dtype=dtype)
        return (xnorm * self.weight).to(dtype=dtype)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.use_nGPT:
            print("normalized")
        else:
            print("NOT normalized")

        self.input_proj = nn.Linear(config.input_dim, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config, il) for il in range(config.n_layer)])

        # Final projection layer
        self.proj = nn.Linear(
            config.n_embd, config.output_dim, bias=config.bias, dtype=torch.bfloat16
        )

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=config.base_scale / math.sqrt(2 * config.n_layer)
                )

        if config.use_nGPT == 1:
            self.sz_init_value = 1.00
            self.sz_init_scaling = config.base_scale
            self.sz = torch.nn.Parameter(
                self.sz_init_scaling * torch.ones(config.output_dim, dtype=torch.float32)
            )

        if config.use_nGPT == 0:
            self.rmsnorm_f = RMSNorm(config.n_embd)

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, particles, features), got {x.dim()}D")
        if x.size(-1) != self.config.input_dim:
            raise ValueError(f"Expected input dim {self.config.input_dim}, got {x.size(-1)}")

        # x shape: (batch_size, num_particles, input_dim)
        b, p, _ = x.size()

        # Project input to embedding dimension
        x = self.input_proj(x)
        x = self.drop(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization if using standard transformer
        if self.config.use_nGPT == 0:
            x = self.rmsnorm_f(x)

        # Final projection
        x = self.proj(x)

        # Scale output if using nGPT
        if self.config.use_nGPT == 1:
            sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
            x = sz.unsqueeze(0).unsqueeze(0) * x

        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
