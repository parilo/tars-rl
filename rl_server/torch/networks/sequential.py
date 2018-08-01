import torch
import torch.nn as nn
from collections import OrderedDict
from itertools import tee


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class SequentialNet(nn.Module):
    def __init__(
            self, hiddens,
            layer_fn=nn.Linear,
            activation_fn=torch.nn.ELU,
            norm_fn=None, bias=True):
        super().__init__()

        net = []

        for i, (f_in, f_out) in enumerate(pairwise(hiddens)):
            net.append((f"layer_{i}", layer_fn(f_in, f_out, bias=bias)))
            if norm_fn is not None:
                net.append((f"norm_{i}", norm_fn(f_out)))
            if activation_fn is not None:
                net.append((f"activation_{i}", activation_fn()))

        self.net = torch.nn.Sequential(OrderedDict(net))

    def forward(self, x):
        x = self.net.forward(x)
        return x
