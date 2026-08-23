from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np


GRAPH_PATTERN = re.compile(r"graph_(\d{4})\.npz$")


class DynamicGraphUnavailableError(ValueError):
    """Raised when a prediction sample has no causally available graph."""


@dataclass(frozen=True)
class GraphAssignment:
    target_index: int
    schedule_slot: int
    graph_slot: int
    graph_time_index: int
    graph_path: Path


def asymmetric_normalize(adjacency: np.ndarray) -> np.ndarray:
    """Return D^-1 A using the Graph WaveNet directed-support convention."""
    adjacency = np.asarray(adjacency, dtype=np.float32)
    row_sum = adjacency.sum(axis=-1)
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum > 0
    inverse[nonzero] = 1.0 / row_sum[nonzero]
    return inverse[..., :, None] * adjacency


class DynamicPCMCIProvider:
    """Map prediction times to the latest causally available PCMCI graph.

    The manifest is the source of truth. For every prediction index, the latest
    schedule time not later than the prediction is selected. Its
    ``assigned_graph_slot`` can point to an earlier valid graph when the current
    PCMCI window was invalid because of missing values. If that exact graph has
    not been generated yet, the sample is rejected instead of silently reusing
    an arbitrarily stale file.
    """

    def __init__(
        self,
        manifest_path: Path | str,
        graph_dir: Path | str,
        adjacency_key: str = "adj_weighted",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.graph_dir = Path(graph_dir)
        self.adjacency_key = adjacency_key

        manifest = np.load(self.manifest_path)
        self.graph_time_indices = manifest["graph_time_indices"].astype(np.int64)
        self.assigned_graph_slots = manifest["assigned_graph_slots"].astype(np.int64)
        if self.graph_time_indices.ndim != 1:
            raise ValueError("graph_time_indices must be one-dimensional")
        if self.graph_time_indices.shape != self.assigned_graph_slots.shape:
            raise ValueError("Manifest graph-time and assignment arrays differ in shape")
        if np.any(np.diff(self.graph_time_indices) <= 0):
            raise ValueError("Manifest graph times must be strictly increasing")

        self._paths: dict[int, Path] = {}
        self._graph_times: dict[int, int] = {}
        self._adjacency_cache: dict[int, np.ndarray] = {}
        for path in sorted(self.graph_dir.glob("graph_*.npz")):
            match = GRAPH_PATTERN.match(path.name)
            if not match:
                continue
            slot = int(match.group(1))
            with np.load(path) as graph:
                graph_time = int(graph["graph_time_index"])
                adjacency = graph[self.adjacency_key]
                if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
                    raise ValueError(f"Invalid adjacency shape in {path}: {adjacency.shape}")
            if slot >= len(self.graph_time_indices):
                raise ValueError(f"Graph slot {slot} is outside the manifest")
            expected_time = int(self.graph_time_indices[slot])
            if graph_time != expected_time:
                raise ValueError(
                    f"Graph {slot} time {graph_time} differs from manifest {expected_time}"
                )
            self._paths[slot] = path
            self._graph_times[slot] = graph_time

        if not self._paths:
            raise DynamicGraphUnavailableError(f"No graph_XXXX.npz files found in {self.graph_dir}")

    @property
    def available_graph_slots(self) -> tuple[int, ...]:
        return tuple(sorted(self._paths))

    def resolve(self, target_indices: Sequence[int] | np.ndarray) -> list[GraphAssignment]:
        targets = np.asarray(target_indices, dtype=np.int64)
        if targets.ndim == 0:
            targets = targets.reshape(1)
        if targets.ndim != 1:
            raise ValueError("target_indices must be a scalar or one-dimensional sequence")

        schedule_slots = np.searchsorted(
            self.graph_time_indices, targets, side="right"
        ) - 1
        assignments: list[GraphAssignment] = []
        failures: list[str] = []
        for target, schedule_slot in zip(targets.tolist(), schedule_slots.tolist()):
            if schedule_slot < 0:
                failures.append(f"target {target}: before the first PCMCI graph")
                continue
            graph_slot = int(self.assigned_graph_slots[schedule_slot])
            if graph_slot < 0:
                failures.append(f"target {target}: no valid historical PCMCI window")
                continue
            path = self._paths.get(graph_slot)
            if path is None:
                failures.append(
                    f"target {target}: required graph_{graph_slot:04d}.npz is not generated"
                )
                continue
            graph_time = self._graph_times[graph_slot]
            if graph_time > target:
                raise AssertionError(
                    f"Future leakage: graph time {graph_time} exceeds target {target}"
                )
            assignments.append(
                GraphAssignment(
                    target_index=int(target),
                    schedule_slot=int(schedule_slot),
                    graph_slot=graph_slot,
                    graph_time_index=graph_time,
                    graph_path=path,
                )
            )

        if failures:
            preview = "; ".join(failures[:5])
            if len(failures) > 5:
                preview += f"; and {len(failures) - 5} more"
            raise DynamicGraphUnavailableError(preview)
        return assignments

    def _load_adjacency(self, graph_slot: int) -> np.ndarray:
        cached = self._adjacency_cache.get(graph_slot)
        if cached is None:
            with np.load(self._paths[graph_slot]) as graph:
                cached = graph[self.adjacency_key].astype(np.float32)
            self._adjacency_cache[graph_slot] = cached
        return cached

    def adjacency_batch(
        self, target_indices: Sequence[int] | np.ndarray
    ) -> tuple[np.ndarray, list[GraphAssignment]]:
        assignments = self.resolve(target_indices)
        batch = np.stack(
            [self._load_adjacency(item.graph_slot) for item in assignments], axis=0
        )
        return batch, assignments

    def normalized_supports(
        self,
        target_indices: Sequence[int] | np.ndarray,
        include_reverse: bool = True,
    ) -> tuple[list[np.ndarray], list[GraphAssignment]]:
        adjacency, assignments = self.adjacency_batch(target_indices)
        supports = [asymmetric_normalize(adjacency)]
        if include_reverse:
            supports.append(asymmetric_normalize(np.swapaxes(adjacency, -1, -2)))
        return supports, assignments

    def torch_supports(
        self,
        target_indices: Sequence[int] | np.ndarray,
        device: object,
        include_reverse: bool = True,
    ) -> tuple[list[object], list[GraphAssignment]]:
        import torch

        supports, assignments = self.normalized_supports(
            target_indices, include_reverse=include_reverse
        )
        return [torch.as_tensor(item, device=device) for item in supports], assignments

    def usable_mask(self, target_indices: Iterable[int]) -> np.ndarray:
        """Return which samples can be used with the graphs generated so far."""
        return np.asarray(
            [self._is_usable(int(target)) for target in target_indices], dtype=bool
        )

    def _is_usable(self, target: int) -> bool:
        schedule_slot = int(
            np.searchsorted(self.graph_time_indices, target, side="right") - 1
        )
        if schedule_slot < 0:
            return False
        assigned = int(self.assigned_graph_slots[schedule_slot])
        return assigned >= 0 and assigned in self._paths and self._graph_times[assigned] <= target
