from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import re
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_GRAPH_DIR = PROJECT_DIR / "dynamic_graphs" / "per_window"
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "cleaned_us101_n_20_full_year"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_DIR / "dynamic_pcmci_window_manifest.npz"
DEFAULT_NODE_DATA_PATH = DEFAULT_DATA_DIR / "full_year_cleaned.npz"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "dynamic_graphs" / "dynamic_graphs.pkl"

GRAPH_PATTERN = re.compile(r"graph_(\d{4})\.npz$")
FORMAT_NAME = "dynamic_causal_adjacency"
FORMAT_VERSION = 1
PICKLE_PROTOCOL = 4  # Compatible with Python 3.6 and newer.


class DynamicGraphExportError(ValueError):
    """Raised when source graph files cannot form a consistent dynamic package."""


def _to_cross_version_pickle_value(value: Any) -> Any:
    """Remove NumPy-specific pickle references while preserving values and shape."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_cross_version_pickle_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cross_version_pickle_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cross_version_pickle_value(item) for item in value)
    return value


def _read_manifest(path: Path) -> dict[str, np.ndarray]:
    required = (
        "window_start_indices",
        "window_end_indices",
        "graph_time_indices",
        "valid_for_pcmci",
        "assigned_graph_slots",
    )
    with np.load(path, allow_pickle=False) as manifest:
        missing = [key for key in required if key not in manifest]
        if missing:
            raise DynamicGraphExportError(
                f"Manifest {path} is missing fields: {', '.join(missing)}"
            )
        arrays = {key: np.asarray(manifest[key]).copy() for key in required}

    schedule_count = len(arrays["graph_time_indices"])
    for key, values in arrays.items():
        if values.ndim != 1 or len(values) != schedule_count:
            raise DynamicGraphExportError(
                f"Manifest field {key} must have shape ({schedule_count},), "
                f"got {values.shape}"
            )
    if np.any(np.diff(arrays["graph_time_indices"].astype(np.int64)) <= 0):
        raise DynamicGraphExportError("Manifest graph times must be strictly increasing")
    return arrays


def _read_node_metadata(path: Path) -> tuple[list[str], list[str]]:
    with np.load(path, allow_pickle=False) as data:
        if "node_ids" not in data:
            raise DynamicGraphExportError(f"Node metadata file {path} has no node_ids")
        node_ids = [str(item) for item in data["node_ids"].tolist()]
        feature_names = (
            [str(item) for item in data["feature_names"].tolist()]
            if "feature_names" in data
            else []
        )
    if not node_ids:
        raise DynamicGraphExportError("node_ids must not be empty")
    if len(set(node_ids)) != len(node_ids):
        raise DynamicGraphExportError("node_ids must be unique")
    return node_ids, feature_names


def _discover_graph_files(graph_dir: Path) -> list[tuple[int, Path]]:
    graph_files: list[tuple[int, Path]] = []
    seen_slots: set[int] = set()
    for path in graph_dir.glob("graph_*.npz"):
        match = GRAPH_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        slot = int(match.group(1))
        if slot in seen_slots:
            raise DynamicGraphExportError(f"Duplicate graph slot {slot} in {graph_dir}")
        seen_slots.add(slot)
        graph_files.append((slot, path))
    graph_files.sort(key=lambda item: item[0])
    if not graph_files:
        raise DynamicGraphExportError(f"No graph_XXXX.npz files found in {graph_dir}")
    return graph_files


def build_dynamic_graph_payload(
    graph_dir: Path | str,
    manifest_path: Path | str,
    node_data_path: Path | str,
    adjacency_key: str = "adj_weighted",
    require_complete: bool = True,
) -> dict[str, Any]:
    """Stack stored per-window matrices without applying an aggregation formula."""
    graph_dir = Path(graph_dir)
    manifest_path = Path(manifest_path)
    node_data_path = Path(node_data_path)

    manifest = _read_manifest(manifest_path)
    node_ids, feature_names = _read_node_metadata(node_data_path)
    graph_files = _discover_graph_files(graph_dir)
    schedule_count = len(manifest["graph_time_indices"])

    graph_slots = np.asarray([slot for slot, _ in graph_files], dtype=np.int64)
    if graph_slots[0] < 0 or graph_slots[-1] >= schedule_count:
        raise DynamicGraphExportError(
            f"Graph slots must be within the manifest range 0..{schedule_count - 1}"
        )

    valid_slots = np.flatnonzero(manifest["valid_for_pcmci"].astype(bool))
    available_slot_set = set(graph_slots.tolist())
    unexpected_slots = sorted(available_slot_set - set(valid_slots.tolist()))
    if unexpected_slots:
        raise DynamicGraphExportError(
            "Graph files exist for manifest-invalid slots: "
            + ", ".join(map(str, unexpected_slots[:10]))
        )
    missing_valid_slots = sorted(set(valid_slots.tolist()) - available_slot_set)
    if require_complete and missing_valid_slots:
        preview = ", ".join(map(str, missing_valid_slots[:10]))
        suffix = "" if len(missing_valid_slots) <= 10 else ", ..."
        raise DynamicGraphExportError(
            f"Missing {len(missing_valid_slots)} valid graph files: {preview}{suffix}"
        )

    matrices: list[np.ndarray] = []
    matrix_shape: tuple[int, int] | None = None
    graph_times: list[int] = []
    for slot, path in graph_files:
        with np.load(path, allow_pickle=False) as graph:
            if adjacency_key not in graph:
                raise DynamicGraphExportError(
                    f"Graph {path} has no stored matrix named {adjacency_key!r}"
                )
            matrix = np.asarray(graph[adjacency_key], dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise DynamicGraphExportError(
                    f"Matrix {adjacency_key!r} in {path} must be square and 2D, "
                    f"got {matrix.shape}"
                )
            if matrix_shape is None:
                matrix_shape = matrix.shape
            elif matrix.shape != matrix_shape:
                raise DynamicGraphExportError(
                    f"Matrix shape changed from {matrix_shape} to {matrix.shape} in {path}"
                )
            if not np.isfinite(matrix).all():
                raise DynamicGraphExportError(f"Matrix {adjacency_key!r} in {path} is not finite")
            if "graph_time_index" not in graph:
                raise DynamicGraphExportError(f"Graph {path} has no graph_time_index")
            graph_time = int(graph["graph_time_index"])

        expected_time = int(manifest["graph_time_indices"][slot])
        if graph_time != expected_time:
            raise DynamicGraphExportError(
                f"Graph slot {slot} time {graph_time} differs from manifest {expected_time}"
            )
        matrices.append(matrix)
        graph_times.append(graph_time)

    assert matrix_shape is not None
    if matrix_shape[0] != len(node_ids):
        raise DynamicGraphExportError(
            f"Matrix node count {matrix_shape[0]} differs from node_ids count {len(node_ids)}"
        )

    assigned_slots = manifest["assigned_graph_slots"].astype(np.int64)
    if np.any(assigned_slots >= schedule_count):
        raise DynamicGraphExportError("Manifest contains an assigned graph slot out of range")
    slot_to_graph_index = {slot: index for index, slot in enumerate(graph_slots.tolist())}
    assigned_graph_indices = np.asarray(
        [slot_to_graph_index.get(int(slot), -1) for slot in assigned_slots],
        dtype=np.int64,
    )
    if require_complete and np.any((assigned_slots >= 0) & (assigned_graph_indices < 0)):
        raise DynamicGraphExportError(
            "At least one scheduled window refers to a graph that was not exported"
        )

    adjacency = np.stack(matrices, axis=0).astype(np.float32, copy=False)
    payload: dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "array_encoding": "python_lists",
        "stored_matrix_key": adjacency_key,
        "orientation": "source_to_target",
        "node_ids": node_ids,
        "node_id_to_index": {node_id: index for index, node_id in enumerate(node_ids)},
        "feature_names": feature_names,
        "adjacency": adjacency,
        "graph_slots": graph_slots,
        "graph_time_indices": np.asarray(graph_times, dtype=np.int64),
        "window_start_indices": manifest["window_start_indices"][graph_slots].astype(np.int64),
        "window_end_indices": manifest["window_end_indices"][graph_slots].astype(np.int64),
        "schedule": {
            "window_start_indices": manifest["window_start_indices"].astype(np.int64),
            "window_end_indices": manifest["window_end_indices"].astype(np.int64),
            "graph_time_indices": manifest["graph_time_indices"].astype(np.int64),
            "valid_for_pcmci": manifest["valid_for_pcmci"].astype(bool),
            "assigned_graph_slots": assigned_slots,
            "assigned_graph_indices": assigned_graph_indices,
        },
    }
    return payload


def save_dynamic_graph_payload(
    payload: dict[str, Any], output_path: Path | str, overwrite: bool = False
) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; use --overwrite to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary_path.open("wb") as handle:
            pickle.dump(
                _to_cross_version_pickle_value(payload),
                handle,
                protocol=PICKLE_PROTOCOL,
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def load_and_validate_dynamic_graph_pickle(path: Path | str) -> dict[str, Any]:
    """Load a trusted export and check its essential structural invariants."""
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise DynamicGraphExportError("Dynamic graph pickle must contain a dictionary")
    if payload.get("format") != FORMAT_NAME or payload.get("format_version") != FORMAT_VERSION:
        raise DynamicGraphExportError("Unsupported dynamic graph pickle format")

    adjacency = np.asarray(payload.get("adjacency"), dtype=np.float32)
    graph_slots = np.asarray(payload.get("graph_slots"), dtype=np.int64)
    graph_times = np.asarray(payload.get("graph_time_indices"), dtype=np.int64)
    node_ids = payload.get("node_ids")
    schedule = payload.get("schedule")
    if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
        raise DynamicGraphExportError(
            f"adjacency must have shape [graphs, nodes, nodes], got {adjacency.shape}"
        )
    if not np.isfinite(adjacency).all():
        raise DynamicGraphExportError("adjacency contains non-finite values")
    if len(graph_slots) != len(adjacency) or len(graph_times) != len(adjacency):
        raise DynamicGraphExportError("Graph metadata length differs from adjacency length")
    if not isinstance(node_ids, list) or len(node_ids) != adjacency.shape[1]:
        raise DynamicGraphExportError("node_ids do not match adjacency node count")
    if not isinstance(schedule, dict):
        raise DynamicGraphExportError("schedule must be a dictionary")
    required_schedule_fields = {
        "window_start_indices",
        "window_end_indices",
        "graph_time_indices",
        "valid_for_pcmci",
        "assigned_graph_slots",
        "assigned_graph_indices",
    }
    missing = required_schedule_fields - set(schedule)
    if missing:
        raise DynamicGraphExportError(
            "schedule is missing fields: " + ", ".join(sorted(missing))
        )
    schedule_lengths = {len(np.asarray(schedule[key])) for key in required_schedule_fields}
    if len(schedule_lengths) != 1:
        raise DynamicGraphExportError("Schedule fields have inconsistent lengths")

    payload["adjacency"] = adjacency
    payload["graph_slots"] = graph_slots
    payload["graph_time_indices"] = graph_times
    payload["window_start_indices"] = np.asarray(
        payload.get("window_start_indices"), dtype=np.int64
    )
    payload["window_end_indices"] = np.asarray(
        payload.get("window_end_indices"), dtype=np.int64
    )
    schedule["window_start_indices"] = np.asarray(
        schedule["window_start_indices"], dtype=np.int64
    )
    schedule["window_end_indices"] = np.asarray(
        schedule["window_end_indices"], dtype=np.int64
    )
    schedule["graph_time_indices"] = np.asarray(
        schedule["graph_time_indices"], dtype=np.int64
    )
    schedule["valid_for_pcmci"] = np.asarray(
        schedule["valid_for_pcmci"], dtype=bool
    )
    schedule["assigned_graph_slots"] = np.asarray(
        schedule["assigned_graph_slots"], dtype=np.int64
    )
    schedule["assigned_graph_indices"] = np.asarray(
        schedule["assigned_graph_indices"], dtype=np.int64
    )
    return payload


def export_dynamic_graphs(
    graph_dir: Path | str,
    manifest_path: Path | str,
    node_data_path: Path | str,
    output_path: Path | str,
    adjacency_key: str = "adj_weighted",
    require_complete: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    payload = build_dynamic_graph_payload(
        graph_dir=graph_dir,
        manifest_path=manifest_path,
        node_data_path=node_data_path,
        adjacency_key=adjacency_key,
        require_complete=require_complete,
    )
    saved_path = save_dynamic_graph_payload(payload, output_path, overwrite=overwrite)
    loaded = load_and_validate_dynamic_graph_pickle(saved_path)
    return {
        "output_path": str(saved_path.resolve()),
        "stored_matrix_key": loaded["stored_matrix_key"],
        "adjacency_shape": list(loaded["adjacency"].shape),
        "adjacency_dtype": str(loaded["adjacency"].dtype),
        "graph_count": int(len(loaded["graph_slots"])),
        "schedule_window_count": int(len(loaded["schedule"]["graph_time_indices"])),
        "unavailable_schedule_count": int(
            np.count_nonzero(loaded["schedule"]["assigned_graph_indices"] < 0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stack matrices already stored in rolling causal graph files and export "
            "one time-indexed pickle package. No causal-weight formula is applied."
        )
    )
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--node-data", type=Path, default=DEFAULT_NODE_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--adjacency-key",
        default="adj_weighted",
        help="Name of the already-stored 2D matrix to export from each graph file",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Export available graphs even when some manifest-valid graph files are missing",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_dynamic_graphs(
        graph_dir=args.graph_dir,
        manifest_path=args.manifest,
        node_data_path=args.node_data,
        output_path=args.output,
        adjacency_key=args.adjacency_key,
        require_complete=not args.allow_incomplete,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
