# Stage 3: Dynamic multivariate PCMCI

This repository contains only the Stage 3 dynamic PCMCI implementation, the
cleaned full-year data, and generated dynamic-graph results. Raw/prepared data,
static PCMCI experiments, Graph WaveNet checkpoints, and duplicate ZIP archives
are intentionally excluded.

## Current configuration

- Source: XTraffic 2023, US-101 northbound corridor.
- Data shape: `(105120, 20, 3)`.
- Features: `flow`, `occupancy`, `speed` (60 PCMCI variables).
- Split: January-August train, September-October validation, November-December test.
- PCMCI history: 7 days (2016 five-minute observations).
- Graph refresh: once per day.
- Lag range: 1-12 steps (5-60 minutes).
- Multiple testing: Benjamini-Hochberg FDR at 0.05.
- Graph WaveNet projection: all source features to target `flow`, maximum absolute significant ParCorr per source-target node pair, strongest 5 incoming source nodes per target.

## Data preparation

`prepare_full_year.py` creates:

- `full_year_cleaned.npz`: model-ready, normalized, PCMCI, and mask arrays;
- `split_indices.npz`: chronological sample indices;
- `scaler.npz`: training-only feature normalization;
- `variable_metadata.csv`: mapping for all 60 variables;
- `missing_runs.csv`: missing-run audit;
- `dynamic_pcmci_window_manifest.*`: rolling-window schedule and validity flags;
- `cleaning_report.json`: complete quality report.

Long gaps remain missing in the PCMCI view, invalidating affected graph windows. The model-ready view fills those gaps only with time-of-day medians learned from the training period. Originally missing target values remain masked for loss and evaluation.

The included `data/cleaned_us101_n_20_full_year` directory is already prepared,
so running `prepare_full_year.py` is not required to generate dynamic graphs.
That script is retained for reproducibility and requires the excluded prepared
source data to be placed in `data/prepared_us101_n_20_full_year`.

## Installation

Python 3.11 is recommended. Create an isolated environment and install the
dependencies:

```powershell
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

## Smoke-test result

The first two daily graphs were generated successfully. Each graph took about 3.5 minutes on the current CPU. Both graphs retained 100 top-k node edges. Their edge Jaccard similarity was 0.5873 and weighted cosine similarity was 0.8911, showing a stable core with measurable daily change.

## Resumable graph generation

Run serially:

```powershell
& '.\.venv\Scripts\python.exe' build_dynamic_pcmci.py
```

Existing graph files are skipped automatically. Four independent workers can partition graph slots without overlap:

```powershell
& '.\.venv\Scripts\python.exe' build_dynamic_pcmci.py --slot-modulus 4 --slot-remainder 0
& '.\.venv\Scripts\python.exe' build_dynamic_pcmci.py --slot-modulus 4 --slot-remainder 1
& '.\.venv\Scripts\python.exe' build_dynamic_pcmci.py --slot-modulus 4 --slot-remainder 2
& '.\.venv\Scripts\python.exe' build_dynamic_pcmci.py --slot-modulus 4 --slot-remainder 3
```

After more than one graph exists, regenerate stability metrics with:

```powershell
& '.\.venv\Scripts\python.exe' analyze_graph_stability.py
```

The current serial estimate for all 341 valid windows is approximately 20 CPU-hours. Parallel speed depends on available physical cores and memory bandwidth.
