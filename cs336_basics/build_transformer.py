import math

import einops
import numpy as np
from jaxtyping import Float, Bool
from torch import nn, Tensor
from torch.nn import init
from torch.nn.init import trunc_normal_
import torch

from cs336_basics.nn_utils import softmax


class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        std = 2.0/(d_in + d_out)
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        self.matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.matrix, mean=0, std=1.0, a=-3.0 * 1.0, b=3.0 * 1.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        self.g = nn.Parameter(torch.empty(d_model, dtype=torch.float32).to(device))
        init.constant_(self.g, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_type = x.dtype
        x = x.to(dtype=torch.float32)
        for b, batch in enumerate(x):
            for s, sequence in enumerate(batch):
                rms: torch.float32
                a_sum: torch.float32 = 0
                for a in sequence:
                    a_sum += a**2
                rms = math.sqrt(a_sum/self.d_model+self.eps)
                for d, a in enumerate(sequence):
                    x[b][s][d] = a/rms
        x = x*self.g
        return x.to(dtype=d_type)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear1(x)
        x = y*torch.sigmoid(y)*self.linear3(x)
        x = self.linear2(x)
        return x


class RoTry(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        thetas = torch.empty(max_seq_len, d_k//2, 2, device=device, dtype=torch.float32)
        for p in range(max_seq_len):
            for k in range(d_k//2):
                theta_k = p*1.0/(theta**((2*k)/d_k))
                thetas[p][k][0] = math.cos(theta_k)
                thetas[p][k][1] = math.sin(theta_k)
        self.register_buffer("thetas", thetas, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        def position_embed(x_b, token_positions_b):
            seq_len = token_positions_b.shape[0]
            x_reshape = einops.rearrange(x_b, "sequence_length (d h) -> sequence_length d h", d=self.d_k // 2, h=2)
            for p in range(seq_len):
                for d in range(self.d_k//2):
                    z = x_reshape[p][d][0]*self.thetas[token_positions_b[p]][d][0]-x_reshape[p][d][1]*self.thetas[token_positions_b[p]][d][1]
                    y = x_reshape[p][d][0]*self.thetas[token_positions_b[p]][d][1]+x_reshape[p][d][1]*self.thetas[token_positions_b[p]][d][0]
                    x_reshape[p][d][0] = z
                    x_reshape[p][d][1] = y
            return einops.rearrange(x_reshape, "sequence_length d h -> sequence_length (d h)")
        print(x.shape)
        print(token_positions.shape)
        if x.dim() == 3:
            batch_size = x.shape[0]
            for b in range(batch_size):
                x[b] = position_embed(x[b], token_positions)
        else:
            x = position_embed(x, token_positions)
        return x

def dot_attention(Q: Float[Tensor, " ... queries d_k"], K: Float[Tensor, " ... keys d_k"], V: Float[Tensor, " ... values d_v"], mask: Bool[Tensor, " ... queries keys"] | None = None):
    K = einops.rearrange(K, "... keys d_k -> ... d_k keys")
    d_k = Q.size(-1)
    score = (Q @ K) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(~mask, -1e9)
    result = softmax(score, dim=-1) @ V
    return result