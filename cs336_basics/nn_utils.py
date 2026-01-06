import numpy as np
import torch


def softmax(in_features, dim):
    x_max = in_features.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(in_features - x_max)
    exp_x /= exp_x.sum(dim = dim, keepdim=True)
    return exp_x