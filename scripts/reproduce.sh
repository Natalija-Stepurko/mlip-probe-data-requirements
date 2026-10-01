#!/usr/bin/env bash
# Re-measure every result from the released dataset.
#
#   scripts/reproduce.sh DATASET_DIR [OUT_DIR] [UMA_DIR]      (N_JOBS=8 by default)
#
# Runs the Magpie and ORB arms, and the UMA arm when the gated UMA release is given (UMA_DIR)
# or found at DATASET_DIR/uma. PYTHON selects the interpreter. Learning curves on 8 cores take
# about 40 min for Magpie, 1 h 20 for ORB and 2 h 20 for UMA; the cross-validation runs take
# many hours. Every step resumes where it stopped.
set -euo pipefail
DATASET=${1:?usage: scripts/reproduce.sh DATASET_DIR [OUT_DIR] [UMA_DIR]}
OUT=${2:-work/results}
UMA=${3:-}
JOBS=${N_JOBS:-8}
RUN="${PYTHON:-python} -m mlip_probe"

ARMS="magpie orb"
if [ -n "$UMA" ] || [ -d "$DATASET/uma" ]; then ARMS="$ARMS uma"; fi
DATA=(--dataset "$DATASET")
if [ -n "$UMA" ]; then DATA+=(--uma-dataset "$UMA"); fi

$RUN ceilings "${DATA[@]}" --out "$OUT"
for arm in $ARMS; do
  for axis in train test; do
    $RUN curves --arm "$arm" --axis "$axis" "${DATA[@]}" --out "$OUT" --n-jobs "$JOBS"
  done
done
$RUN summarise --out "$OUT"
$RUN gain --out "$OUT"
$RUN split-fraction --out "$OUT"
for arm in $ARMS; do
  $RUN cv --arm "$arm" "${DATA[@]}" --out "$OUT" --n-jobs "$JOBS"
done
$RUN cv-summarise --out "$OUT"
echo "done: $OUT"
