from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "cleaned_us101_n_20_full_year"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "dynamic_graphs"
LEGACY_TIGRAMITE_SOURCE = PROJECT_DIR.parent / "tigramite改后" / "tigramite"

TAU_MIN = 1
TAU_MAX = 12
PC_ALPHA = 0.05
ALPHA_LEVEL = 0.05
MAX_CONDS_DIM = 3
MAX_CONDS_MCI = 3
TOP_K_INCOMING = 5
TARGET_FEATURE = "flow"


def benjamini_hochberg_lagged(p_matrix: np.ndarray) -> np.ndarray:
    q_matrix = np.full_like(p_matrix, np.nan, dtype=np.float64)
    lagged_p = p_matrix[:, :, TAU_MIN : TAU_MAX + 1]
    finite = np.isfinite(lagged_p)
    finite_p = lagged_p[finite]
    if not finite_p.size:
        return q_matrix
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    adjusted = ranked * finite_p.size / np.arange(1, finite_p.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q_lagged = q_matrix[:, :, TAU_MIN : TAU_MAX + 1]
    q_lagged[finite] = restored
    return q_matrix


def standardize_window(window: np.ndarray) -> np.ndarray:
    flat = window.reshape(len(window), -1).astype(np.float64)
    if not np.isfinite(flat).all():
        raise ValueError("PCMCI window contains non-finite values")
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    if np.any(stds < 1e-8):
        indices = np.flatnonzero(stds < 1e-8).tolist()
        raise ValueError(f"Near-constant PCMCI variables in this window: {indices}")
    return (flat - means) / stds


def aggregate_to_flow_node_graph(
    p_matrix: np.ndarray,
    q_matrix: np.ndarray,
    val_matrix: np.ndarray,
    node_ids: np.ndarray,
    feature_names: np.ndarray,
    graph_slot: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    node_count = len(node_ids)
    feature_count = len(feature_names)
    flow_matches = np.flatnonzero(feature_names == TARGET_FEATURE)
    if flow_matches.size != 1:
        raise ValueError("Flow target feature is not unique")
    flow_feature = int(flow_matches[0])

    full_binary = np.zeros((node_count, node_count), dtype=np.float32)
    full_weighted = np.zeros_like(full_binary)
    full_signed = np.zeros_like(full_binary)
    edge_rows: list[dict[str, object]] = []

    for source_var in range(node_count * feature_count):
        source_node, source_feature = divmod(source_var, feature_count)
        for target_node in range(node_count):
            target_var = target_node * feature_count + flow_feature
            for lag in range(TAU_MIN, TAU_MAX + 1):
                q_value = float(q_matrix[source_var, target_var, lag])
                if not np.isfinite(q_value) or q_value > ALPHA_LEVEL:
                    continue
                value = float(val_matrix[source_var, target_var, lag])
                edge_rows.append(
                    {
                        "graph_slot": graph_slot,
                        "source_node_index": source_node,
                        "source_node_id": str(node_ids[source_node]),
                        "source_feature": str(feature_names[source_feature]),
                        "target_node_index": target_node,
                        "target_node_id": str(node_ids[target_node]),
                        "target_feature": TARGET_FEATURE,
                        "lag_5min_steps": lag,
                        "raw_p_value": float(p_matrix[source_var, target_var, lag]),
                        "fdr_bh_q_value": q_value,
                        "parcorr_value": value,
                    }
                )
                if source_node == target_node:
                    continue
                full_binary[source_node, target_node] = 1.0
                if abs(value) > full_weighted[source_node, target_node]:
                    full_weighted[source_node, target_node] = abs(value)
                    full_signed[source_node, target_node] = value

    sparse_binary = np.zeros_like(full_binary)
    sparse_weighted = np.zeros_like(full_weighted)
    sparse_signed = np.zeros_like(full_signed)
    for target_node in range(node_count):
        candidates = np.flatnonzero(full_weighted[:, target_node] > 0)
        if not candidates.size:
            continue
        ordered = candidates[np.argsort(full_weighted[candidates, target_node])[::-1]]
        keep = ordered[:TOP_K_INCOMING]
        sparse_binary[keep, target_node] = 1.0
        sparse_weighted[keep, target_node] = full_weighted[keep, target_node]
        sparse_signed[keep, target_node] = full_signed[keep, target_node]

    return {
        "significant_binary": full_binary,
        "significant_weighted": full_weighted,
        "adj_binary": sparse_binary,
        "adj_weighted": sparse_weighted,
        "adj_signed": sparse_signed,
    }, edge_rows


def run_one_graph(
    graph_slot: int,
    window: np.ndarray,
    node_ids: np.ndarray,
    feature_names: np.ndarray,
    graph_dir: Path,
    window_start_index: int,
    window_end_index: int,
    graph_time_index: int,
) -> dict[str, object]:
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ModuleNotFoundError:
        if not LEGACY_TIGRAMITE_SOURCE.exists():
            raise ModuleNotFoundError(
                "Tigramite is not installed. Run: pip install -r requirements.txt"
            )
        sys.path.insert(0, str(LEGACY_TIGRAMITE_SOURCE))
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI

    standardized = standardize_window(window)
    variable_names = [
        f"{node_ids[node]}|{feature_names[feature]}"
        for node in range(len(node_ids))
        for feature in range(len(feature_names))
    ]
    dataframe = pp.DataFrame(standardized, var_names=variable_names)
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    started = time.perf_counter()
    results = pcmci.run_pcmci(
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
        pc_alpha=PC_ALPHA,
        max_conds_dim=MAX_CONDS_DIM,
        max_combinations=1,
        max_conds_py=MAX_CONDS_MCI,
        max_conds_px=MAX_CONDS_MCI,
        alpha_level=ALPHA_LEVEL,
        fdr_method="none",
    )
    elapsed = time.perf_counter() - started
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    q_matrix = benjamini_hochberg_lagged(p_matrix)
    graphs, edge_rows = aggregate_to_flow_node_graph(
        p_matrix, q_matrix, val_matrix, node_ids, feature_names, graph_slot
    )

    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / f"graph_{graph_slot:04d}.npz"
    np.savez_compressed(
        graph_path,
        **graphs,
        p_matrix=p_matrix.astype(np.float32),
        q_matrix=q_matrix.astype(np.float32),
        val_matrix=val_matrix.astype(np.float32),
        window_start_index=np.asarray(window_start_index, dtype=np.int64),
        window_end_index=np.asarray(window_end_index, dtype=np.int64),
        graph_time_index=np.asarray(graph_time_index, dtype=np.int64),
    )
    edge_path = graph_dir / f"graph_{graph_slot:04d}_edges.csv"
    edge_fields = [
        "graph_slot",
        "source_node_index",
        "source_node_id",
        "source_feature",
        "target_node_index",
        "target_node_id",
        "target_feature",
        "lag_5min_steps",
        "raw_p_value",
        "fdr_bh_q_value",
        "parcorr_value",
    ]
    with edge_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=edge_fields)
        writer.writeheader()
        writer.writerows(edge_rows)

    finite_tests = int(np.count_nonzero(np.isfinite(p_matrix[:, :, TAU_MIN : TAU_MAX + 1])))
    return {
        "graph_slot": graph_slot,
        "runtime_seconds": elapsed,
        "finite_lagged_tests": finite_tests,
        "significant_flow_target_variable_edges": len(edge_rows),
        "significant_node_edges": int(graphs["significant_binary"].sum()),
        "top_k_node_edges": int(graphs["adj_binary"].sum()),
        "graph_path": str(graph_path),
        "edge_path": str(edge_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rolling multivariate PCMCI graphs")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-slot", type=int, default=0)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--slot-modulus", type=int, default=1)
    parser.add_argument("--slot-remainder", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.slot_modulus < 1:
        raise ValueError("--slot-modulus must be at least 1")
    if args.slot_remainder < 0 or args.slot_remainder >= args.slot_modulus:
        raise ValueError("--slot-remainder must be in [0, slot-modulus)")
    clean = np.load(args.data_dir / "full_year_cleaned.npz")
    pcmci_data = clean["data_pcmci"]
    node_ids = clean["node_ids"].astype(str)
    feature_names = clean["feature_names"].astype(str)
    timestamps = clean["timestamps"].astype(str)
    manifest = np.load(args.data_dir / "dynamic_pcmci_window_manifest.npz")
    starts = manifest["window_start_indices"]
    ends = manifest["window_end_indices"]
    graph_times = manifest["graph_time_indices"]
    valid = manifest["valid_for_pcmci"]

    graph_dir = args.output_dir / "per_window"
    summaries = []
    attempted = 0
    for slot in range(args.start_slot, len(starts)):
        if slot % args.slot_modulus != args.slot_remainder:
            continue
        if not bool(valid[slot]):
            continue
        output_path = graph_dir / f"graph_{slot:04d}.npz"
        if output_path.exists() and not args.overwrite:
            continue
        if args.max_graphs is not None and attempted >= args.max_graphs:
            break
        attempted += 1
        start = int(starts[slot])
        end = int(ends[slot])
        graph_time = int(graph_times[slot])
        print(
            f"Graph slot {slot}: {timestamps[start]} to {timestamps[end - 1]} "
            f"-> {timestamps[graph_time]}",
            flush=True,
        )
        summary = run_one_graph(
            slot,
            pcmci_data[start:end],
            node_ids,
            feature_names,
            graph_dir,
            start,
            end,
            graph_time,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "data_dir": str(args.data_dir),
        "variable_count": int(len(node_ids) * len(feature_names)),
        "target_feature": TARGET_FEATURE,
        "tau_min": TAU_MIN,
        "tau_max": TAU_MAX,
        "pc_alpha": PC_ALPHA,
        "fdr_alpha": ALPHA_LEVEL,
        "top_k_incoming_nodes": TOP_K_INCOMING,
        "completed_this_run": summaries,
    }
    (args.output_dir / "latest_run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
