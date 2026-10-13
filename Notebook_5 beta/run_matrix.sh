#!/usr/bin/env bash
# Experiment matrix for the NIH vs reference-notebook gap. Appends to results/runs.jsonl.
# Usage: PY=../.gpu/Scripts/python.exe bash run_matrix.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
SEEDS="42 43 44 45 46"
run() { "$PY" -W ignore gap_experiments.py "$@"; }

# E1: reference protocol replayed on NIH (Infiltration-only vs No Finding)
run --tag E1_reference_only --positive only --seeds $SEEDS
# E2: reference label definition (any Infiltration, co-findings allowed)
run --tag E2_reference_any --positive any --seeds $SEEDS
# E3: label-shuffle control; AUC must stay near 0.5
run --tag E3_shuffled_labels --positive only --shuffle-labels --seeds 42 43 44
# E4: v4-style frozen backbone on the same small data
run --tag E4_layer4_only --positive only --trainable layer4 --seeds 42 43 44
run --tag E4_head_only --positive only --trainable head --seeds 42 43 44
# E5: more data, full fine-tune, scored on the whole filtered official test set
run --tag E5_full_ft_2000 --positive only --train-per-class 2000 --val-per-class 200 \
    --test-per-class 0 --seeds 42
run --tag E5_full_ft_2000_any --positive any --train-per-class 2000 --val-per-class 200 \
    --test-per-class 0 --seeds 42
