# Extracted for the Mona-only KernelAdapter export, 2026-10-14.
"""Mona spatial adapter, retaining the existing projection and convolution layout."""
import torch
from torch import nn
from torch.nn import functional as F


class MonaAdapter(nn.Module):
    def __init__(self, dim, width=64):
        super().__init__()
        self.width = width
        self.project1 = nn.Linear(dim, width)
        self.project2 = nn.Linear(width, dim)
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-6))
        self.gammax = nn.Parameter(torch.ones(dim))
        self.dropout = nn.Dropout(.1)
        self.convs = nn.ModuleList([nn.Conv2d(width, width, k, padding=k//2, groups=width)
                                    for k in (3, 5, 7)])
        self.projector = nn.Conv2d(width, width, 1)
        # Match Mona's global init_weights policy on its linear/LN modules.
        for layer in (self.project1, self.project2):
            nn.init.trunc_normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)

    def encode(self, x, hw):
        z = self.project1(self.norm(x)*self.gamma + x*self.gammax)
        z = z.reshape(x.shape[0], *hw, self.width).permute(0, 3, 1, 2)
        z = z + sum(conv(z) for conv in self.convs)/3.
        z = z + self.projector(z)
        return z.permute(0, 2, 3, 1)

    def decode(self, v):
        return self.project2(self.dropout(F.gelu(v))).flatten(1, 2)

    def forward(self, x, hw):
        return x + self.decode(self.encode(x, hw))
