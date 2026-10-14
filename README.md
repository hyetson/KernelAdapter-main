# KernelAdapter

Core PyTorch implementation of KernelAdapter with Mona adapters on a frozen Swin backbone.
A low-rank reader accesses pre-compression activations at MLP fc2 and patch-merging reductions;
a GELU writer adds output corrections. Default: rank 8, Mona width 64.

## Installation

Python 3.10+. Install the appropriate CPU/CUDA PyTorch build for your machine.

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

## Usage

```python
import torch
from kerneladapter import KernelSwin

model = KernelSwin(pretrained="/path/to/swin_base_patch4_window7_224_22k.pth")
model.init_weights()  # load frozen weights, then initialize kernel readers
features = model(torch.randn(1, 3, 224, 224))  # four BCHW feature maps
auxiliary_losses = model.auxiliary_losses()
```

Train Mona and KernelAdapter parameters with your task head. Add the sum of
`auxiliary_losses.values()` to the task loss (kernel weight 0.01, orthogonality weight 0.001).
Use `pretrained=None` for a random-weight smoke check, not pretrained transfer.
This export contains model components and checkpoint utilities only.

## License

Apache-2.0; third-party attribution is retained in `NOTICE`.
