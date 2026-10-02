#!/usr/bin/env bash
# Learned-tier dependencies + weights (video/photo tiers, damage classifier).
# depth-anything-3 pins numpy<2 and xformers (no macOS wheel); it runs fine without both, so
# it is installed with --no-deps and its real runtime deps are listed explicitly.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY -m pip install -q --no-deps depth-anything-3==0.1.1
$PY -m pip install -q einops omegaconf safetensors imageio plyfile trimesh e3nn typer pillow-heif addict moviepy==1.0.3 evo
export HF_HOME="$PWD/weights/hf"
$PY - <<'EOF'
import sys, types
for m in ("pycolmap", "moviepy", "moviepy.editor"):
    sys.modules.setdefault(m, types.ModuleType(m))
from huggingface_hub import snapshot_download
for rid in ["depth-anything/DA3METRIC-LARGE", "depth-anything/DA3-BASE", "openai/clip-vit-base-patch32"]:
    snapshot_download(rid); print("fetched", rid)
import kornia.feature as KF
KF.LoFTR(pretrained="indoor_new"); print("fetched LoFTR indoor_new")
EOF
