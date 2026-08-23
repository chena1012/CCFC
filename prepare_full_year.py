from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "data" / "prepared_us101_n_20_full_year"
OUTPUT_DIR = PROJECT_DIR / "data" / "cleaned_us101_n_20_full_year"
JANUARY_NODE_IDS = SOURCE_DIR / "selected_nodes.csv"

TRAIN_END_TIME = np.datetime64("2023-09-01T00:00:00")
VAL_END_TIME = np.datetime64("2023-11-01T00:00:00")
EXPECTED_STEP_SECONDS = 300
STEPS_PER_DAY = 288
SHORT_GAP_LIMIT = 12
PCMCI_WINDOW_STEPS = 7 * STEPS_PER_DAY
PCMCI_UPDATE_STEPS = STEPS_PER_DAY
HISTORY = 12
HORIZON = 12
TARGET_FEATURE = "flow"


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def read_authoritative_node_ids() -> list[str]:
    with JANUARY_NODE_IDS.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["node_index"]))
    return [str(row.get("node_id") or row["station_id"]) for row in rows]


def causal_short_gap_fill(
    data: np.ndarray, fallback_by_node_feature: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    filled = data.copy()
    short_filled_mask = np.zeros(data.shape, dtype=bool)
    for node in range(data.shape[1]):
        for feature in range(data.shape[2]):
            values = filled[:, node, feature]
            missing = ~np.isfinite(values)
            for start, end in contiguous_runs(missing):
                if end - start > SHORT_GAP_LIMIT:
                    continue
                if start > 0 and np.isfinite(values[start - 1]):
                    replacement = float(values[start - 1])
                else:
                    replacement = float(fallback_by_node_feature[node, feature])
                values[start:end] = replacement
                short_filled_mask[start:end, node, feature] = True
    return filled, short_filled_mask


def training_time_slot_medians(
    data: np.ndarray, train_end: int, fallback: np.ndarray
) -> np.ndarray:
    medians = np.empty((STEPS_PER_DAY, data.shape[1], data.shape[2]), dtype=np.float32)
    for slot in range(STEPS_PER_DAY):
        slot_values = data[slot:train_end:STEPS_PER_DAY]
        with np.errstate(all="ignore"):
            median = np.nanmedian(slot_values, axis=0)
        median = np.where(np.isfinite(median), median, fallback)
        medians[slot] = median.astype(np.float32)
    return medians


def fill_remaining_from_training_pattern(
    data: np.ndarray, slot_medians: np.ndarray, fallback: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    filled = data.copy()
    imputed_mask = ~np.isfinite(filled)
    for time_index in range(len(filled)):
        missing = ~np.isfinite(filled[time_index])
        if not np.any(missing):
            continue
        replacement = slot_medians[time_index % STEPS_PER_DAY]
        replacement = np.where(np.isfinite(replacement), replacement, fallback)
        filled[time_index][missing] = replacement[missing]
    if not np.isfinite(filled).all():
        raise ValueError("Model-ready data still contains non-finite values")
    return filled, imputed_mask


def split_target_indices(length: int, train_end: int, val_end: int) -> dict[str, np.ndarray]:
    bounds = {
        "train": (HISTORY, train_end),
        "val": (train_end, val_end),
        "test": (val_end, length),
    }
    result = {}
    for name, (start, stop) in bounds.items():
        result[name] = np.arange(max(start, HISTORY), stop - HORIZON + 1, dtype=np.int64)
    return result


def build_pcmci_manifest(
    timestamps: np.ndarray, pcmci_data: np.ndarray
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    window_starts = []
    window_ends = []
    graph_times = []
    valid_flags = []
    assigned_valid_slots = []
    latest_valid = -1

    for graph_slot, graph_time_index in enumerate(
        range(PCMCI_WINDOW_STEPS, len(timestamps), PCMCI_UPDATE_STEPS)
    ):
        window_start = graph_time_index - PCMCI_WINDOW_STEPS
        window_end = graph_time_index
        window = pcmci_data[window_start:window_end]
        missing_cells = int(np.count_nonzero(~np.isfinite(window)))
        valid = missing_cells == 0
        if valid:
            latest_valid = graph_slot

        rows.append(
            {
                "graph_slot": graph_slot,
                "graph_time": str(timestamps[graph_time_index]),
                "window_start": str(timestamps[window_start]),
                "window_end_exclusive": str(timestamps[window_end]),
                "window_steps": PCMCI_WINDOW_STEPS,
                "missing_cells": missing_cells,
                "valid_for_pcmci": int(valid),
                "assigned_graph_slot": latest_valid,
            }
        )
        window_starts.append(window_start)
        window_ends.append(window_end)
        graph_times.append(graph_time_index)
        valid_flags.append(valid)
        assigned_valid_slots.append(latest_valid)

    arrays = {
        "window_start_indices": np.asarray(window_starts, dtype=np.int64),
        "window_end_indices": np.asarray(window_ends, dtype=np.int64),
        "graph_time_indices": np.asarray(graph_times, dtype=np.int64),
        "valid_for_pcmci": np.asarray(valid_flags, dtype=bool),
        "assigned_graph_slots": np.asarray(assigned_valid_slots, dtype=np.int64),
    }
    return rows, arrays


def main() -> None:
    source_path = SOURCE_DIR / "xtraffic_2023_subgraph.npz"
    source = np.load(source_path)
    raw = source["data"].astype(np.float32)
    timestamps_text = source["timestamps"].astype(str)
    timestamps = timestamps_text.astype("datetime64[s]")
    node_ids = source["node_ids"].astype(str)
    feature_names = source["feature_names"].astype(str)

    expected_node_ids = read_authoritative_node_ids()
    if node_ids.tolist() != expected_node_ids:
        raise ValueError("Full-year node order differs from the January authoritative order")
    if raw.shape != (365 * STEPS_PER_DAY, 20, 3):
        raise ValueError(f"Unexpected full-year shape: {raw.shape}")
    if len(np.unique(timestamps)) != len(timestamps):
        raise ValueError("Duplicate timestamps detected")
    deltas = np.diff(timestamps).astype("timedelta64[s]").astype(np.int64)
    if not np.all(deltas == EXPECTED_STEP_SECONDS):
        raise ValueError("Non-five-minute timestamp interval detected")
    if feature_names.tolist() != ["flow", "occupancy", "speed"]:
        raise ValueError(f"Unexpected feature order: {feature_names.tolist()}")

    train_end = int(np.searchsorted(timestamps, TRAIN_END_TIME))
    val_end = int(np.searchsorted(timestamps, VAL_END_TIME))
    if timestamps[train_end] != TRAIN_END_TIME or timestamps[val_end] != VAL_END_TIME:
        raise ValueError("Calendar split boundary is absent")

    observed_mask = np.isfinite(raw) & (raw >= 0)
    invalid = ~observed_mask
    cleaned = raw.copy()
    cleaned[invalid] = np.nan

    with np.errstate(all="ignore"):
        fallback = np.nanmedian(cleaned[:train_end], axis=0)
    if not np.isfinite(fallback).all():
        raise ValueError("Training period cannot provide a finite fallback for every variable")

    pcmci_data, short_filled_mask = causal_short_gap_fill(cleaned, fallback)
    long_gap_mask = ~np.isfinite(pcmci_data)
    slot_medians = training_time_slot_medians(pcmci_data, train_end, fallback)
    model_data, long_imputed_mask = fill_remaining_from_training_pattern(
        pcmci_data, slot_medians, fallback
    )

    means = model_data[:train_end].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    stds = model_data[:train_end].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)
    normalized = ((model_data - means) / stds).astype(np.float32)

    target_feature_matches = np.flatnonzero(feature_names == TARGET_FEATURE)
    if target_feature_matches.size != 1:
        raise ValueError("Target feature is not unique")
    target_feature = int(target_feature_matches[0])
    target_observed = observed_mask[:, :, target_feature]
    target_indices = split_target_indices(len(raw), train_end, val_end)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_DIR / "full_year_cleaned.npz",
        data_model_ready=model_data,
        data_normalized=normalized,
        data_pcmci=pcmci_data,
        observed_mask=observed_mask,
        short_filled_mask=short_filled_mask,
        long_gap_mask=long_gap_mask,
        target_observed_mask=target_observed,
        timestamps=timestamps_text,
        node_ids=node_ids,
        feature_names=feature_names,
    )
    np.savez_compressed(
        OUTPUT_DIR / "scaler.npz",
        mean=means,
        std=stds,
        feature_names=feature_names,
        target_feature_index=np.asarray(target_feature, dtype=np.int64),
    )
    np.savez_compressed(
        OUTPUT_DIR / "split_indices.npz",
        train_target_indices=target_indices["train"],
        val_target_indices=target_indices["val"],
        test_target_indices=target_indices["test"],
        train_end=np.asarray(train_end, dtype=np.int64),
        val_end=np.asarray(val_end, dtype=np.int64),
    )

    manifest_rows, manifest_arrays = build_pcmci_manifest(timestamps_text, pcmci_data)
    with (OUTPUT_DIR / "dynamic_pcmci_window_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    np.savez_compressed(OUTPUT_DIR / "dynamic_pcmci_window_manifest.npz", **manifest_arrays)

    with (OUTPUT_DIR / "variable_metadata.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["variable_index", "node_index", "node_id", "feature_index", "feature"])
        variable_index = 0
        for node_index, node_id in enumerate(node_ids):
            for feature_index, feature_name in enumerate(feature_names):
                writer.writerow(
                    [variable_index, node_index, str(node_id), feature_index, str(feature_name)]
                )
                variable_index += 1

    missing_runs = []
    for node, node_id in enumerate(node_ids):
        for feature, feature_name in enumerate(feature_names):
            for start, end in contiguous_runs(invalid[:, node, feature]):
                missing_runs.append(
                    {
                        "node_index": node,
                        "node_id": str(node_id),
                        "feature": str(feature_name),
                        "start": str(timestamps_text[start]),
                        "end": str(timestamps_text[end - 1]),
                        "steps": end - start,
                        "minutes": (end - start) * 5,
                        "short_gap_filled": int(end - start <= SHORT_GAP_LIMIT),
                    }
                )
    with (OUTPUT_DIR / "missing_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(missing_runs[0]))
        writer.writeheader()
        writer.writerows(missing_runs)

    valid_graphs = int(sum(row["valid_for_pcmci"] for row in manifest_rows))
    report = {
        "source": str(source_path),
        "shape": list(raw.shape),
        "timestamp_start": str(timestamps_text[0]),
        "timestamp_end": str(timestamps_text[-1]),
        "cadence_seconds": EXPECTED_STEP_SECONDS,
        "node_count": int(raw.shape[1]),
        "feature_names": feature_names.tolist(),
        "variable_count": int(raw.shape[1] * raw.shape[2]),
        "split": {
            "train": [str(timestamps_text[0]), str(timestamps_text[train_end - 1])],
            "val": [str(timestamps_text[train_end]), str(timestamps_text[val_end - 1])],
            "test": [str(timestamps_text[val_end]), str(timestamps_text[-1])],
        },
        "missing_cells_original": int(np.count_nonzero(invalid)),
        "short_gap_cells_filled_for_pcmci": int(np.count_nonzero(short_filled_mask)),
        "long_gap_cells_remaining_for_pcmci": int(np.count_nonzero(long_gap_mask)),
        "long_gap_cells_imputed_for_model": int(np.count_nonzero(long_imputed_mask)),
        "model_ready_nonfinite_cells": int(np.count_nonzero(~np.isfinite(model_data))),
        "normalization_fitted_on": "training period only",
        "normalization_mean_by_feature": dict(zip(feature_names.tolist(), means.tolist())),
        "normalization_std_by_feature": dict(zip(feature_names.tolist(), stds.tolist())),
        "pcmci_schedule": {
            "history_days": 7,
            "window_steps": PCMCI_WINDOW_STEPS,
            "update_hours": 24,
            "candidate_graph_slots": len(manifest_rows),
            "valid_graph_slots": valid_graphs,
            "invalid_graph_slots": len(manifest_rows) - valid_graphs,
            "invalid_slots_reuse_latest_valid_graph": True,
        },
        "target_feature": TARGET_FEATURE,
        "target_missing_values_are_masked_in_loss_and_evaluation": True,
    }
    (OUTPUT_DIR / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    shutil.copy2(SOURCE_DIR / "road_adj.npy", OUTPUT_DIR / "road_adj.npy")
    shutil.copy2(SOURCE_DIR / "selected_nodes.csv", OUTPUT_DIR / "selected_nodes.csv")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
