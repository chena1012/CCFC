from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_graph_provider import DynamicPCMCIProvider


class BatchAwareNConv(nn.Module):
    """Graph WaveNet nconv supporting shared or per-sample adjacency matrices."""

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim == 2:
            output = torch.einsum("ncvl,vw->ncwl", x, adjacency)
        elif adjacency.ndim == 3:
            if adjacency.shape[0] != x.shape[0]:
                raise ValueError(
                    "Batched adjacency and input have different batch sizes: "
                    f"{adjacency.shape[0]} != {x.shape[0]}"
                )
            output = torch.einsum("ncvl,nvw->ncwl", x, adjacency)
        else:
            raise ValueError(
                f"Adjacency must have shape [V,V] or [B,V,V], got {adjacency.shape}"
            )
        return output.contiguous()


def install_batch_aware_nconv(model: nn.Module) -> int:
    """Replace Graph WaveNet's nconv modules without changing its source file."""
    replacements = 0
    for module in model.modules():
        if hasattr(module, "nconv") and isinstance(module.nconv, nn.Module):
            if not isinstance(module.nconv, BatchAwareNConv):
                module.nconv = BatchAwareNConv()
                replacements += 1
    if replacements == 0:
        raise ValueError("No Graph WaveNet nconv modules were found in the model")
    return replacements


def upgrade_legacy_spatiotemporal_convs(model: nn.Module) -> int:
    """Convert legacy 4-D Conv1d weights to Conv2d for modern PyTorch.

    The original Graph WaveNet repository constructs several ``Conv1d`` layers
    with two-dimensional kernel tuples. Older environments tolerated this, but
    modern PyTorch rejects their 4-D input. The parameter shapes are already
    Conv2d-compatible, so this conversion preserves the weights.
    """
    replacements = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if not isinstance(child, nn.Conv1d) or child.weight.ndim != 4:
                continue
            replacement = nn.Conv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            ).to(device=child.weight.device, dtype=child.weight.dtype)
            replacement.weight = child.weight
            replacement.bias = child.bias
            setattr(parent, name, replacement)
            replacements += 1
    return replacements


class DynamicGraphWaveNetAdapter(nn.Module):
    """Run an existing Graph WaveNet with one causal PCMCI graph per sample.

    Construct the original model with two placeholder static supports. The
    adapter replaces those supports only for the duration of each forward pass,
    then restores them. Adaptive adjacency remains supported because the
    original model appends it internally.
    """

    def __init__(
        self,
        model: nn.Module,
        provider: DynamicPCMCIProvider,
        include_reverse: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.provider = provider
        self.include_reverse = include_reverse
        expected_supports = 2 if include_reverse else 1
        current_supports = getattr(model, "supports", None)
        if current_supports is None or len(current_supports) != expected_supports:
            raise ValueError(
                "Construct Graph WaveNet with exactly "
                f"{expected_supports} placeholder support matrix/matrices; its GCN "
                "channel sizes are fixed during initialization."
            )
        upgrade_legacy_spatiotemporal_convs(model)
        install_batch_aware_nconv(model)
        self.last_assignments = []

    def forward(
        self, inputs: torch.Tensor, target_indices: Sequence[int] | np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(target_indices, torch.Tensor):
            target_indices = target_indices.detach().cpu().numpy()
        supports, assignments = self.provider.torch_supports(
            target_indices,
            device=inputs.device,
            include_reverse=self.include_reverse,
        )
        if len(assignments) != inputs.shape[0]:
            raise ValueError("Every input sample must have exactly one graph assignment")

        previous_supports = self.model.supports
        self.model.supports = supports
        try:
            output = self.model(inputs)
        finally:
            self.model.supports = previous_supports
        self.last_assignments = assignments
        return output

    def forward_history_batch(
        self,
        history: torch.Tensor,
        target_indices: Sequence[int] | np.ndarray | torch.Tensor,
        left_padding: int = 1,
    ) -> torch.Tensor:
        """Accept dataset batches shaped [B, history, nodes, features]."""
        if history.ndim != 4:
            raise ValueError(
                "history must have shape [batch, history, nodes, features]"
            )
        model_input = history.permute(0, 3, 2, 1).contiguous()
        if left_padding:
            model_input = F.pad(model_input, (left_padding, 0, 0, 0))
        return self(model_input, target_indices)


class FullYearGraphWaveNetDataset(torch.utils.data.Dataset):
    """Chronological Graph WaveNet samples with their prediction-time indices."""

    def __init__(
        self,
        cleaned_data_path: Path | str,
        split_indices_path: Path | str,
        split: str,
        history: int = 12,
        horizon: int = 12,
        target_feature: str = "flow",
        provider: DynamicPCMCIProvider | None = None,
        only_generated_graphs: bool = False,
    ) -> None:
        cleaned = np.load(Path(cleaned_data_path))
        split_data = np.load(Path(split_indices_path))
        key = f"{split}_target_indices"
        if key not in split_data.files:
            raise ValueError(f"Unknown split {split!r}; expected train, val, or test")

        self.inputs = cleaned["data_normalized"].astype(np.float32)
        self.targets = cleaned["data_model_ready"].astype(np.float32)
        feature_names = cleaned["feature_names"].astype(str)
        matches = np.flatnonzero(feature_names == target_feature)
        if matches.size != 1:
            raise ValueError(f"Target feature {target_feature!r} is not unique")
        self.target_feature_index = int(matches[0])
        self.history = int(history)
        self.horizon = int(horizon)
        self.target_indices = split_data[key].astype(np.int64)

        in_bounds = (self.target_indices >= self.history) & (
            self.target_indices + self.horizon <= len(self.inputs)
        )
        self.target_indices = self.target_indices[in_bounds]
        if only_generated_graphs:
            if provider is None:
                raise ValueError("provider is required when only_generated_graphs=True")
            self.target_indices = self.target_indices[
                provider.usable_mask(self.target_indices)
            ]

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        target_index = int(self.target_indices[position])
        x = self.inputs[target_index - self.history : target_index]
        y = self.targets[
            target_index : target_index + self.horizon,
            :,
            self.target_feature_index,
        ]
        return torch.from_numpy(x), torch.from_numpy(y), target_index
