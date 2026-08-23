from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
GRAPH_DIR = PROJECT_DIR / "dynamic_graphs" / "per_window"
OUTPUT_DIR = PROJECT_DIR / "dynamic_graphs"
GRAPH_PATTERN = re.compile(r"graph_(\d{4})\.npz$")


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(left * right) / denominator)


def main() -> None:
    graph_files = []
    for path in GRAPH_DIR.glob("graph_*.npz"):
        match = GRAPH_PATTERN.match(path.name)
        if match:
            graph_files.append((int(match.group(1)), path))
    graph_files.sort()
    if len(graph_files) < 2:
        raise ValueError("At least two generated graphs are required")

    rows = []
    previous_slot, previous_path = graph_files[0]
    previous = np.load(previous_path)
    previous_binary = previous["adj_binary"].astype(bool)
    previous_weighted = previous["adj_weighted"].astype(np.float64)

    for slot, path in graph_files[1:]:
        current = np.load(path)
        current_binary = current["adj_binary"].astype(bool)
        current_weighted = current["adj_weighted"].astype(np.float64)
        intersection = int(np.count_nonzero(previous_binary & current_binary))
        union = int(np.count_nonzero(previous_binary | current_binary))
        previous_edges = int(np.count_nonzero(previous_binary))
        current_edges = int(np.count_nonzero(current_binary))
        added = int(np.count_nonzero(~previous_binary & current_binary))
        removed = int(np.count_nonzero(previous_binary & ~current_binary))
        previous_norm = float(np.linalg.norm(previous_weighted))
        relative_change = (
            float(np.linalg.norm(current_weighted - previous_weighted) / previous_norm)
            if previous_norm > 0
            else float("nan")
        )
        rows.append(
            {
                "previous_graph_slot": previous_slot,
                "current_graph_slot": slot,
                "slot_gap": slot - previous_slot,
                "previous_edges": previous_edges,
                "current_edges": current_edges,
                "intersection_edges": intersection,
                "union_edges": union,
                "added_edges": added,
                "removed_edges": removed,
                "edge_jaccard": intersection / union if union else float("nan"),
                "weighted_cosine": cosine_similarity(previous_weighted, current_weighted),
                "relative_frobenius_change": relative_change,
            }
        )
        previous_slot = slot
        previous_binary = current_binary
        previous_weighted = current_weighted

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "graph_stability.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "generated_graph_count": len(graph_files),
        "consecutive_pairs": len(rows),
        "mean_edge_jaccard": float(np.mean([row["edge_jaccard"] for row in rows])),
        "mean_weighted_cosine": float(np.mean([row["weighted_cosine"] for row in rows])),
        "mean_relative_frobenius_change": float(
            np.mean([row["relative_frobenius_change"] for row in rows])
        ),
        "pairs": rows,
    }
    (OUTPUT_DIR / "graph_stability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
