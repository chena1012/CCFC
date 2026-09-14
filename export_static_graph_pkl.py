from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from export_dynamic_graphs_pkl import load_and_validate_dynamic_graph_pickle


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DYNAMIC_PICKLE = PROJECT_DIR / "dynamic_graphs" / "dynamic_graphs.pkl"
DEFAULT_SPLIT_INDICES = (
    PROJECT_DIR
    / "data"
    / "cleaned_us101_n_20_full_year"
    / "split_indices.npz"
)
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "dynamic_graphs" / "static_pcmci_exp.pkl"
DEFAULT_METADATA_PATH = (
    PROJECT_DIR / "dynamic_graphs" / "static_pcmci_exp.metadata.json"
)
PICKLE_PROTOCOL = 2  # Matches the original Graph WaveNet/DCRNN adjacency files.


class StaticGraphExportError(ValueError):
    """Raised when a leakage-free static graph cannot be built."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_last_target_index(split_indices_path: Path | str) -> int:
    split_indices_path = Path(split_indices_path)
    with np.load(split_indices_path, allow_pickle=False) as split:
        if "train_target_indices" not in split:
            raise StaticGraphExportError(
                f"Split file {split_indices_path} has no train_target_indices"
            )
        train_targets = np.asarray(split["train_target_indices"], dtype=np.int64)
    if train_targets.ndim != 1 or not train_targets.size:
        raise StaticGraphExportError("train_target_indices must be a non-empty vector")
    if np.any(np.diff(train_targets) <= 0):
        raise StaticGraphExportError("train_target_indices must be strictly increasing")
    return int(train_targets[-1])


def _top_k_incoming(adjacency: np.ndarray, top_k: int | None) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.float32).copy()
    np.fill_diagonal(adjacency, 0.0)
    if top_k is None:
        return adjacency
    if top_k < 1:
        raise StaticGraphExportError("top_k must be at least 1, or omitted")

    sparse = np.zeros_like(adjacency)
    for target_node in range(adjacency.shape[1]):
        candidates = np.flatnonzero(adjacency[:, target_node] > 0)
        if not candidates.size:
            continue
        order = candidates[
            np.argsort(adjacency[candidates, target_node], kind="stable")[::-1]
        ]
        keep = order[:top_k]
        sparse[keep, target_node] = adjacency[keep, target_node]
    return sparse


def build_exponentially_weighted_static_adjacency(
    dynamic_payload: dict[str, Any],
    train_last_target_index: int,
    decay_lambda: float,
    max_history_lag: int | None = None,
    top_k_incoming: int | None = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aggregate training-time graph matrices using the supplied decay formula.

    The newest eligible schedule graph has age k=0. If max_history_lag is K,
    the formula uses K+1 matrices (the current matrix plus K earlier matrices).
    When max_history_lag is None, every training-time schedule matrix is used.
    """
    if not np.isfinite(decay_lambda) or decay_lambda < 0:
        raise StaticGraphExportError("decay_lambda must be finite and non-negative")
    if max_history_lag is not None and max_history_lag < 0:
        raise StaticGraphExportError("max_history_lag must be non-negative or omitted")

    adjacency = np.asarray(dynamic_payload["adjacency"], dtype=np.float32)
    graph_slots = np.asarray(dynamic_payload["graph_slots"], dtype=np.int64)
    schedule = dynamic_payload["schedule"]
    schedule_times = np.asarray(schedule["graph_time_indices"], dtype=np.int64)
    assigned_graph_slots = np.asarray(
        schedule["assigned_graph_slots"], dtype=np.int64
    )
    assigned_graph_indices = np.asarray(
        schedule["assigned_graph_indices"], dtype=np.int64
    )

    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise StaticGraphExportError("Dynamic adjacency must have shape [graphs, nodes, nodes]")
    if np.any(adjacency < 0):
        raise StaticGraphExportError(
            "Exponential static export expects non-negative stored graph matrices"
        )
    if schedule_times.shape != assigned_graph_indices.shape:
        raise StaticGraphExportError("Schedule time and graph assignment lengths differ")
    if assigned_graph_slots.shape != assigned_graph_indices.shape:
        raise StaticGraphExportError("Schedule graph slot and index lengths differ")

    eligible_positions = np.flatnonzero(schedule_times <= int(train_last_target_index))
    if not eligible_positions.size:
        raise StaticGraphExportError("No causal graph is available within the training period")
    if not np.array_equal(
        eligible_positions,
        np.arange(eligible_positions[-1] + 1, dtype=np.int64),
    ):
        raise StaticGraphExportError("Training-eligible graph schedules must form a prefix")

    training_graph_indices = assigned_graph_indices[eligible_positions]
    if np.any(training_graph_indices < 0):
        missing_positions = eligible_positions[training_graph_indices < 0]
        raise StaticGraphExportError(
            "Training schedules refer to unavailable graphs at positions: "
            + ", ".join(map(str, missing_positions[:10].tolist()))
        )
    if np.any(training_graph_indices >= len(adjacency)):
        raise StaticGraphExportError("A training schedule graph index is out of range")

    training_graphs = adjacency[training_graph_indices]
    available_history_lag = len(training_graphs) - 1
    effective_history_lag = (
        available_history_lag
        if max_history_lag is None
        else min(max_history_lag, available_history_lag)
    )
    selected_graphs = training_graphs[-(effective_history_lag + 1) :]
    ages = np.arange(effective_history_lag, -1, -1, dtype=np.float64)
    unnormalized_weights = np.exp(-float(decay_lambda) * ages)
    normalized_weights = unnormalized_weights / unnormalized_weights.sum()
    dense_static = np.tensordot(
        normalized_weights, selected_graphs.astype(np.float64), axes=(0, 0)
    ).astype(np.float32)
    np.fill_diagonal(dense_static, 0.0)
    static_adjacency = _top_k_incoming(dense_static, top_k_incoming)

    selected_positions = eligible_positions[-(effective_history_lag + 1) :]
    selected_slots = assigned_graph_slots[selected_positions]
    metadata: dict[str, Any] = {
        "formula": (
            "A_static = sum_{k=0}^K exp(-lambda*k) * G_(T-k) "
            "/ sum_{k=0}^K exp(-lambda*k)"
        ),
        "decay_lambda": float(decay_lambda),
        "half_life_windows": (
            float(math.log(2.0) / decay_lambda) if decay_lambda > 0 else None
        ),
        "requested_max_history_lag_K": max_history_lag,
        "effective_max_history_lag_K": int(effective_history_lag),
        "matrix_count_K_plus_1": int(effective_history_lag + 1),
        "top_k_incoming": top_k_incoming,
        "train_last_target_index": int(train_last_target_index),
        "first_selected_schedule_position": int(selected_positions[0]),
        "last_selected_schedule_position": int(selected_positions[-1]),
        "first_selected_graph_time_index": int(schedule_times[selected_positions[0]]),
        "last_selected_graph_time_index": int(schedule_times[selected_positions[-1]]),
        "first_selected_graph_slot": int(selected_slots[0]),
        "last_selected_graph_slot": int(selected_slots[-1]),
        "stored_matrix_key": str(dynamic_payload.get("stored_matrix_key", "unknown")),
        "orientation": str(dynamic_payload.get("orientation", "source_to_target")),
        "node_count": int(static_adjacency.shape[0]),
        "nonzero_edges_before_top_k": int(np.count_nonzero(dense_static)),
        "nonzero_edges_after_top_k": int(np.count_nonzero(static_adjacency)),
        "normalization": (
            "Not stored in the pkl; Graph WaveNet load_adj applies the requested "
            "normalization such as doubletransition."
        ),
    }
    return static_adjacency, metadata


def save_graph_wavenet_static_pickle(
    adjacency: np.ndarray,
    node_ids: list[str],
    node_id_to_index: dict[str, int],
    output_path: Path | str,
    overwrite: bool = False,
) -> Path:
    """Save the original Graph WaveNet three-item adjacency structure."""
    output_path = Path(output_path)
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise StaticGraphExportError("Static adjacency must be a square 2D matrix")
    if adjacency.shape[0] != len(node_ids):
        raise StaticGraphExportError("Static adjacency and node_ids sizes differ")
    expected_mapping = {node_id: index for index, node_id in enumerate(node_ids)}
    if node_id_to_index != expected_mapping:
        raise StaticGraphExportError("node_id_to_index does not match node_ids order")
    if not np.isfinite(adjacency).all():
        raise StaticGraphExportError("Static adjacency contains non-finite values")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use --overwrite")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    graph_wavenet_payload = (
        list(node_ids),
        {str(key): int(value) for key, value in node_id_to_index.items()},
        adjacency.tolist(),
    )
    try:
        with temporary_path.open("wb") as handle:
            pickle.dump(graph_wavenet_payload, handle, protocol=PICKLE_PROTOCOL)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def load_and_validate_graph_wavenet_static_pickle(
    path: Path | str,
) -> tuple[list[str], dict[str, int], np.ndarray]:
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, (list, tuple)) or len(payload) != 3:
        raise StaticGraphExportError(
            "Graph WaveNet adjacency pickle must contain three items"
        )
    node_ids, node_id_to_index, adjacency = payload
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if not isinstance(node_ids, list) or not isinstance(node_id_to_index, dict):
        raise StaticGraphExportError("Invalid node metadata in static adjacency pickle")
    if adjacency.shape != (len(node_ids), len(node_ids)):
        raise StaticGraphExportError("Static adjacency shape does not match node_ids")
    if not np.isfinite(adjacency).all():
        raise StaticGraphExportError("Static adjacency contains non-finite values")
    return node_ids, node_id_to_index, adjacency


def export_static_graph(
    dynamic_pickle_path: Path | str,
    split_indices_path: Path | str,
    output_path: Path | str,
    metadata_path: Path | str,
    decay_lambda: float,
    max_history_lag: int | None = None,
    top_k_incoming: int | None = 5,
    overwrite: bool = False,
) -> dict[str, Any]:
    dynamic_pickle_path = Path(dynamic_pickle_path)
    split_indices_path = Path(split_indices_path)
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use --overwrite")
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Metadata already exists: {metadata_path}; use --overwrite")

    dynamic_payload = load_and_validate_dynamic_graph_pickle(dynamic_pickle_path)
    train_last_target = _training_last_target_index(split_indices_path)
    adjacency, metadata = build_exponentially_weighted_static_adjacency(
        dynamic_payload=dynamic_payload,
        train_last_target_index=train_last_target,
        decay_lambda=decay_lambda,
        max_history_lag=max_history_lag,
        top_k_incoming=top_k_incoming,
    )
    save_graph_wavenet_static_pickle(
        adjacency=adjacency,
        node_ids=dynamic_payload["node_ids"],
        node_id_to_index=dynamic_payload["node_id_to_index"],
        output_path=output_path,
        overwrite=overwrite,
    )
    _, _, verified_adjacency = load_and_validate_graph_wavenet_static_pickle(
        output_path
    )
    if not np.array_equal(adjacency, verified_adjacency):
        raise StaticGraphExportError("Saved adjacency differs from the computed matrix")

    metadata.update(
        {
            "output_path": str(output_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
            "source_dynamic_pickle": str(dynamic_pickle_path.resolve()),
            "source_dynamic_pickle_sha256": _sha256(dynamic_pickle_path),
            "output_pickle_sha256": _sha256(output_path),
            "adjacency_shape": list(adjacency.shape),
            "adjacency_dtype": str(adjacency.dtype),
            "minimum_weight": float(adjacency.min()),
            "maximum_weight": float(adjacency.max()),
        }
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata_path = metadata_path.with_name(metadata_path.name + ".tmp")
    try:
        temporary_metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_metadata_path.replace(metadata_path)
    finally:
        if temporary_metadata_path.exists():
            temporary_metadata_path.unlink()
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate training-only dynamic PCMCI matrices into one exponentially "
            "weighted static Graph WaveNet adjacency pickle."
        )
    )
    parser.add_argument("--dynamic-pkl", type=Path, default=DEFAULT_DYNAMIC_PICKLE)
    parser.add_argument("--split-indices", type=Path, default=DEFAULT_SPLIT_INDICES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_PATH)
    decay_group = parser.add_mutually_exclusive_group(required=True)
    decay_group.add_argument("--decay-lambda", type=float)
    decay_group.add_argument(
        "--half-life-windows",
        type=float,
        help="Set lambda=ln(2)/half_life_windows",
    )
    parser.add_argument(
        "--max-history-lag",
        type=int,
        default=None,
        help="K in the formula; omit to use every training-time schedule matrix",
    )
    parser.add_argument("--top-k-incoming", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.half_life_windows is not None:
        if not np.isfinite(args.half_life_windows) or args.half_life_windows <= 0:
            raise StaticGraphExportError("half_life_windows must be positive and finite")
        decay_lambda = math.log(2.0) / args.half_life_windows
    else:
        decay_lambda = args.decay_lambda
    assert decay_lambda is not None

    metadata = export_static_graph(
        dynamic_pickle_path=args.dynamic_pkl,
        split_indices_path=args.split_indices,
        output_path=args.output,
        metadata_path=args.metadata_output,
        decay_lambda=decay_lambda,
        max_history_lag=args.max_history_lag,
        top_k_incoming=args.top_k_incoming,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
