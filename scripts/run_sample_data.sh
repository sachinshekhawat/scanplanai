#!/usr/bin/env bash
# Run every sample capture through every tier, then the benchmark, and copy the light-weight
# outputs (plan.json/png/svg/log) into results/sample_data/ for review.
#
#   scripts/run_sample_data.sh [ZIP_DIR=SampleDataTotest] [TAG=final]
#
# LiDAR tier reads the Stray zip directly. Video tier uses only the zip's rgb.mp4. Photo tier
# uses per-room still folders derived from the video (scripts/make_photo_set.py, disclosed in
# each MANIFEST.txt) because the sample data contains no photo sets.
set -euo pipefail
cd "$(dirname "$0")/.."
ZIP_DIR="${1:-SampleDataTotest}"
TAG="${2:-final}"
PY=${PY:-.venv/bin/python}
scripts/fetch_data.sh "$ZIP_DIR"
$PY -m scan2plan.benchmark.run --tag "$TAG"
OUT=results/sample_data
rm -rf "$OUT" && mkdir -p "$OUT"
for d in runs/bench_"$TAG"/*/; do
  n=$(basename "$d"); mkdir -p "$OUT/$n"
  cp "$d"/plan.json "$d"/plan.png "$d"/plan.svg "$d"/log.txt "$OUT/$n/" 2>/dev/null || true
done
cp -r data/photo_sets "$OUT/photo_sets_manifests" 2>/dev/null && find "$OUT/photo_sets_manifests" -name '*.jpg' -delete
echo "outputs in $OUT, report in benchmark/results/$TAG/report.md"
