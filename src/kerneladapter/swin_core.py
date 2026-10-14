# Extracted for the Mona-only KernelAdapter export, 2026-10-14.
"""Spatial Swin primitives adapted from HiRA-L (Apache-2.0)."""
import torch
from torch import nn
from torch.nn import functional as F
def tokens_to_grid(x,hw):
    return x.reshape(x.shape[0],*hw,x.shape[-1])
def grid_to_tokens(x):
    return x.flatten(1,2)

def drop_path(inputs: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
    if probability == 0.0 or not training:
        return inputs
    keep = 1.0 - probability
    shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
    random = keep + torch.rand(shape, dtype=inputs.dtype, device=inputs.device)
    return inputs * random.floor() / keep

class DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return drop_path(inputs, self.probability, self.training)

def window_partition(grid: torch.Tensor, window_size: int) -> torch.Tensor:
    batch, height, width, channels = grid.shape
    if height % window_size or width % window_size:
        raise ValueError("window_partition requires padded spatial dimensions")
    grid = grid.reshape(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return grid.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)

def window_reverse(
    windows: torch.Tensor, window_size: int, height: int, width: int, batch: int
) -> torch.Tensor:
    channels = windows.shape[-1]
    grid = windows.reshape(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    return grid.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels)

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(inputs)))))

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if images.ndim != 4:
            raise ValueError("PatchEmbedding expects BCHW images")
        pad_h = (self.patch_size - images.shape[-2] % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - images.shape[-1] % self.patch_size) % self.patch_size
        images = F.pad(images, (0, pad_w, 0, pad_h))
        features = self.projection(images)
        height, width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        return self.norm(tokens), (height, width)

class PatchMerging(nn.Module):
    def __init__(self, dim: int, output_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, output_dim, bias=False)

    def forward(self, tokens: torch.Tensor, hw_shape: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
        grid = tokens_to_grid(tokens, hw_shape)
        height, width = hw_shape
        if height % 2 or width % 2:
            grid = F.pad(grid, (0, 0, 0, width % 2, 0, height % 2))
        x0 = grid[:, 0::2, 0::2, :]
        x1 = grid[:, 1::2, 0::2, :]
        x2 = grid[:, 0::2, 1::2, :]
        x3 = grid[:, 1::2, 1::2, :]
        merged = torch.cat((x0, x1, x2, x3), dim=-1)
        merged = self.reduction(self.norm(merged))
        return grid_to_tokens(merged), (merged.shape[1], merged.shape[2])
