import math

import einops
from torch import nn
from torch.nn import init
from torch.nn.init import trunc_normal_
import torch


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