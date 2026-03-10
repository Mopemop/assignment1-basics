import numpy as np
import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math


def softmax(in_features, dim):
    x_max = in_features.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(in_features - x_max)
    exp_x /= exp_x.sum(dim = dim, keepdim=True)
    return exp_x

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = softmax(inputs, 1)
    inputs = inputs[torch.arange(inputs.shape[0]), targets]
    loss = -torch.log(inputs + 1e-8)
    return loss.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                grad = p.grad.data
                m = state.get("m", 0)
                v = state.get("v", 0)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.pow(grad, 2)
                lrt = lr*math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lrt * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1