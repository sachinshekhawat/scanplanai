#!/usr/bin/env bash
# Unpack raw Stray Scanner captures into data/raw/<capture_name>/<scan_id>/.
# Usage: scripts/fetch_data.sh [ZIP_DIR]   (default: SampleDataTotest)
set -euo pipefail
ZIP_DIR="${1:-SampleDataTotest}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data/raw"
shopt -s nullglob
zips=("$ZIP_DIR"/*.zip)
if [ ${#zips[@]} -eq 0 ]; then
  echo "No zips found in $ZIP_DIR" >&2; exit 1
fi
for z in "${zips[@]}"; do
  name="$(basename "$z" .zip)"
  dest="$ROOT/data/raw/$name"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest")" ]; then
    echo "skip $name (already unpacked)"; continue
  fi
  mkdir -p "$dest"
  echo "unpacking $name ..."
  unzip -q -o "$z" -d "$dest"
done
echo "done: $(ls "$ROOT/data/raw")"
