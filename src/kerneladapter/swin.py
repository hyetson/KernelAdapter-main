# Extracted for the Mona-only KernelAdapter export, 2026-10-14.
"""Mona-backed KernelAdapter at Swin MLP and patch-merging compression sites."""

from dataclasses import asdict

from .backbone import MonaSwin

from .discarded import AdaptedLinear, KernelConfig, auxiliary_losses, initialize, operators


class KernelSwin(MonaSwin):
    def __init__(
        self,
        discarded_mode="full",
        discarded_space="kernel",
        discarded_rank=8,
        discarded_activation="gelu",
        kernel_loss_weight=0.01,
        basis_orth_weight=0.001,
        projection_eps=1e-6,
        **kwargs,
    ):
        if discarded_mode not in {"none", "mlp", "merge", "full"}:
            raise ValueError("Unknown discarded_mode")
        super().__init__(**kwargs)
        self.discarded_mode = discarded_mode
        cfg = KernelConfig(
            discarded_rank,
            discarded_space,
            discarded_activation,
            kernel_loss_weight,
            basis_orth_weight,
            projection_eps,
        )
        if discarded_mode in {"mlp", "full"}:
            for stage in self.stages:
                for block in stage.blocks:
                    block.mlp.fc2 = AdaptedLinear.from_linear(block.mlp.fc2, cfg)
        if discarded_mode in {"merge", "full"}:
            for merger in self.mergers:
                merger.reduction = AdaptedLinear.from_linear(merger.reduction, cfg)
        self.kernel_signature = {
            "mode": discarded_mode,
            "kernel": asdict(cfg),
            "channels": list(self.channels),
            "depths": list(self.depths),
            "base": self.base,
            "base_width": self.base_width,
        }
        self.apply_trainable_policy()

    def apply_trainable_policy(self):
        super().apply_trainable_policy()
        for _, op in operators(self):
            op.discarded.up.requires_grad_(True)
            op.discarded.basis.requires_grad_(True)

    def init_weights(self):
        super().init_weights()
        initialize(self)

    def auxiliary_losses(self):
        return auxiliary_losses(self)
