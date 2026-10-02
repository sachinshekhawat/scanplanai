"""Child-process Depth Anything 3 inference (torch only; pycolmap stubbed: libomp clash).

    python -m scan2plan.learned.da3_worker <list.txt> <out.npz> [model_id] [process_res]

<list.txt>: one image path per line. Writes depth (N,H,W) f16, conf (N,H,W) f16,
extrinsics (N,3,4) w2c OpenCV, intrinsics (N,3,3) at processed resolution, and the
processed image size. Model weights are fetched from Hugging Face into weights/hf.
"""
from __future__ import annotations

import os
import sys
import time
import types

for _m in ("pycolmap", "moviepy", "moviepy.editor"):
    sys.modules.setdefault(_m, types.ModuleType(_m))

import numpy as np  # noqa: E402

DEFAULT_MODEL = "depth-anything/DA3-BASE"


FOCAL_PRIOR = 0.83  # iPhone 1x main camera: f ~ 0.83 x long side (69 deg HFOV, 4:3)


def metric_main(lst: str, out: str, res: int = 504):
    """Monocular metric depth, DA3METRIC-LARGE: depth_m = focal_px * net / 300 (model card).
    Output resized to 256x192 (LiDAR depth grid) to share the fusion code."""
    import cv2
    import torch
    from depth_anything_3.api import DepthAnything3

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to(device=dev)
    paths = [p.strip() for p in open(lst) if p.strip()]
    outd = {}
    t = time.time()
    for p in paths:
        with torch.no_grad():
            pr = model.inference([p], process_res=res)
        d = pr.depth[0].astype(np.float32)
        f = FOCAL_PRIOR * max(d.shape)
        outd[os.path.splitext(os.path.basename(p))[0]] = cv2.resize(f * d / 300.0, (256, 192)).astype(np.float16)
    np.savez(out, **outd)
    print(f"[da3] metric depth on {len(paths)} images in {time.time() - t:.0f} s ({dev})", flush=True)


def _patch_attention():
    import torch
    import torch.nn.functional as F

    _sdpa = F.scaled_dot_product_attention

    def chunked_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        """Query-chunked attention: MPS materialises the full (Lq x Lk) matrix, which for
        multi-view global attention (N views x ~1000 tokens) exceeds unified memory."""
        Lq = q.shape[-2]
        step = max(1, int(2.0e8 // max(1, q.shape[0] * q.shape[1] * k.shape[-2])))
        if Lq <= step:
            return _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        outs = []
        for s in range(0, Lq, step):
            m = attn_mask[..., s:s + step, :] if attn_mask is not None and attn_mask.dim() >= 2 else attn_mask
            outs.append(_sdpa(q[..., s:s + step, :], k, v, attn_mask=m, dropout_p=dropout_p, scale=scale))
        return torch.cat(outs, dim=-2)

    F.scaled_dot_product_attention = chunked_sdpa


def multi_main(jobs_json: str, res: int = 504):
    """Several pose/depth windows with one model load: jobs = [[list.txt, out.npz], ...]."""
    import json

    import torch
    from depth_anything_3.api import DepthAnything3

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained(DEFAULT_MODEL).to(device=dev)
    t = time.time()
    jobs = json.load(open(jobs_json))
    for lst, out in jobs:
        paths = [p.strip() for p in open(lst) if p.strip()]
        try:
            with torch.no_grad():
                pred = model.inference(paths, process_res=res)
        except Exception as e:  # one bad window must not kill the others
            print(f"[da3] window {lst} failed: {e!r}", flush=True)
            continue
        np.savez(out, depth=pred.depth.astype(np.float16), conf=pred.conf.astype(np.float16),
                 extrinsics=pred.extrinsics.astype(np.float32), intrinsics=pred.intrinsics.astype(np.float32),
                 hw=np.array(pred.depth.shape[1:]), names=np.array([os.path.basename(p) for p in paths]))
    print(f"[da3] {len(jobs)} windows in {time.time() - t:.0f} s ({dev})", flush=True)


def main():
    _patch_attention()
    if sys.argv[1] == "multi":
        os.environ.setdefault("HF_HOME", os.path.abspath("weights/hf"))
        return multi_main(sys.argv[2])
    if sys.argv[1] == "metric":
        os.environ.setdefault("HF_HOME", os.path.abspath("weights/hf"))
        return metric_main(sys.argv[2], sys.argv[3])
    lst, out = sys.argv[1], sys.argv[2]
    model_id = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_MODEL
    res = int(sys.argv[4]) if len(sys.argv) > 4 else 504
    os.environ.setdefault("HF_HOME", os.path.abspath("weights/hf"))
    import torch
    from depth_anything_3.api import DepthAnything3

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    paths = [p.strip() for p in open(lst) if p.strip()]
    t = time.time()
    model = DepthAnything3.from_pretrained(model_id).to(device=dev)
    t_load = time.time() - t
    t = time.time()
    with torch.no_grad():
        pred = model.inference(paths, process_res=res, use_ray_pose=os.environ.get("DA3_RAY_POSE", "0") == "1")
    dt = time.time() - t
    np.savez(out, depth=pred.depth.astype(np.float16), conf=pred.conf.astype(np.float16),
             extrinsics=None if pred.extrinsics is None else pred.extrinsics.astype(np.float32),
             intrinsics=None if pred.intrinsics is None else pred.intrinsics.astype(np.float32),
             hw=np.array(pred.depth.shape[1:]), names=np.array([os.path.basename(p) for p in paths]))
    print(f"[da3] {model_id} on {len(paths)} images at res {res}: load {t_load:.0f} s, infer {dt:.0f} s ({dev})",
          flush=True)


if __name__ == "__main__":
    main()
