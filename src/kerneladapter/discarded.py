# Extracted for the Mona-only KernelAdapter export, 2026-10-14.
"""Low-rank kernel reader, nonlinear writer and soft alignment losses."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class KernelConfig:
    rank: int = 8
    space: str = "kernel"
    activation: str = "gelu"
    kernel_loss_weight: float = 0.01
    basis_orth_weight: float = 0.001
    projection_eps: float = 1e-6

    def __post_init__(self):
        if self.rank < 1 or self.space != "kernel":
            raise ValueError("Positive rank and kernel space required")
        if self.activation != "gelu":
            raise ValueError("This export uses the nonlinear GELU writer")
        if min(self.kernel_loss_weight, self.basis_orth_weight) < 0 or self.projection_eps <= 0:
            raise ValueError("Loss weights must be nonnegative; projection_eps positive")


class DiscardedSubspaceAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, config):
        super().__init__()
        if in_dim <= out_dim or config.rank > in_dim - out_dim:
            raise ValueError("Compression requires di>do and rank<=di-do")
        self.config = config
        self.in_dim, self.out_dim = in_dim, out_dim
        self.rank = config.rank
        self.basis = nn.Parameter(torch.empty(in_dim, self.rank))
        self.up = nn.Linear(self.rank, out_dim, bias=False)
        nn.init.normal_(self.basis, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.enabled = True

    @torch.no_grad()
    def initialize_from_frozen_weight(self, weight):
        if weight.requires_grad:
            raise ValueError("Compression weight must be frozen before initialization")
        with torch.autocast(device_type=weight.device.type, enabled=False):
            w = weight.detach().float()
            r = torch.randn_like(self.basis, dtype=torch.float32)
            gram = w @ w.T
            # Relative ridge makes initialization insensitive to weight scale.
            ridge = self.config.projection_eps * gram.diagonal().mean().clamp_min(1e-12)
            row = w.T @ torch.linalg.solve(
                gram + ridge * torch.eye(w.shape[0], device=w.device), w @ r
            )
            r = r - row
            q, triangular = torch.linalg.qr(r, mode="reduced")
            if triangular.diagonal().abs().min() < 1e-8:
                raise ValueError("Initialization failed: insufficient numerical subspace rank")
            self.basis.copy_(q.to(self.basis))
            self.up.weight.zero_()

    def forward(self, hidden):
        coeff = hidden @ self.basis
        coeff = F.gelu(coeff)
        return self.up(coeff)

    def statistics(self, weight):
        with torch.autocast(device_type=weight.device.type, enabled=False):
            w, b = weight.detach().float(), self.basis.float()
            orth = (b.T @ b - torch.eye(self.rank, device=b.device)).square().mean()
            leakage = (
                self.in_dim
                * (w @ b).square().sum()
                / (w.square().sum() * b.square().sum() + 1e-12)
            )
            return leakage, orth

    def losses(self, weight):
        leakage, orth = self.statistics(weight)
        return leakage * self.config.kernel_loss_weight, orth * self.config.basis_orth_weight



class AdaptedLinear(nn.Linear):
    """Keep original weight/bias keys and inject before downstream dropout/residual."""

    @classmethod
    def from_linear(cls, linear, config):
        obj = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        obj.weight = linear.weight
        obj.bias = linear.bias
        obj.weight.requires_grad_(False)
        if obj.bias is not None:
            obj.bias.requires_grad_(False)
        obj.discarded = DiscardedSubspaceAdapter(
            linear.in_features, linear.out_features, config
        ).to(linear.weight)
        return obj

    def forward(self, hidden):
        base = F.linear(hidden, self.weight, self.bias)
        if not self.discarded.enabled:
            return base
        return base + self.discarded(hidden)


def operators(model):
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, AdaptedLinear)
    ]


def initialize(model):
    for _, module in operators(model):
        module.discarded.initialize_from_frozen_weight(module.weight)


def auxiliary_losses(model):
    pairs = [module.discarded.losses(module.weight) for _, module in operators(model)]
    if not pairs:
        return {}
    return {
        "loss_kernel": sum(p[0] for p in pairs) / len(pairs),
        "loss_kernel_basis_orthogonal": sum(p[1] for p in pairs) / len(pairs),
    }
