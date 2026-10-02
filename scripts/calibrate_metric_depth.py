"""Measure the metric-depth scale bias of DA3METRIC-LARGE against iPhone LiDAR.

For each Stray capture: N frames spread over the walk, rotated upright (Stray stores frames
sideways), metric depth with the iPhone focal prior, compared with high-confidence LiDAR depth
(0.3-4 m). Reports per-capture median ratio and spread; the pooled median is DA_SCALE_BIAS
used by the video and photo tiers, the spread across frames feeds their scale sigma.

    python scripts/calibrate_metric_depth.py [N=24]  -> benchmark/results/metric_depth_calibration.json
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scan2plan.io.stray import load_stray  # noqa: E402
from scan2plan.video.pipeline import _unrotate_depths  # noqa: E402


def main(n=24):
    out = {}
    pooled = []
    for d in sorted((ROOT / "data/raw").iterdir()):
        cap = load_stray(d)
        work = ROOT / "data/cache/calib" / d.name
        work.mkdir(parents=True, exist_ok=True)
        vc = cv2.VideoCapture(str(cap.root / "rgb.mp4"))
        idx = np.linspace(50, cap.n_frames - 50, n).astype(int)
        paths = []
        for i in idx:
            vc.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = vc.read()
            p = work / f"{i:06d}.jpg"
            cv2.imwrite(str(p), cv2.rotate(cv2.resize(f, (1024, 768)), cv2.ROTATE_90_CLOCKWISE))
            paths.append(str(p))
        (work / "list.txt").write_text("\n".join(paths))
        npz = work / "depth.npz"
        if not npz.exists():
            subprocess.run([sys.executable, "-m", "scan2plan.learned.da3_worker", "metric", str(work / "list.txt"), str(npz)],
                           check=True, capture_output=True)
        z = np.load(npz)
        dd = _unrotate_depths({k: z[k].astype(np.float32) for k in z.files}, 1)
        r, a = [], []
        for i in idx:
            dm = dd[f"{i:06d}"]
            g = cap.depth(int(i))
            c = cap.confidence(int(i))
            m = (c == 2) & (g > 0.3) & (g < 4)
            if m.sum() < 500:
                continue
            k = float(np.median(dm[m] / g[m]))
            r.append(k)
            a.append(float(np.median(np.abs(dm[m] / k - g[m]) / g[m])))
        r = np.array(r)
        pooled += r.tolist()
        out[d.name] = {"n": int(len(r)), "ratio_median": float(np.median(r)),
                       "ratio_iqr": [float(x) for x in np.percentile(r, [25, 75])],
                       "per_frame_log_sigma": float(np.median(np.abs(np.log(r / np.median(r)))) * 1.4826),
                       "absrel_after_scale": float(np.median(a))}
        print(d.name, out[d.name])
    pooled = np.array(pooled)
    out["pooled"] = {"ratio_median": float(np.median(pooled)),
                     "per_frame_log_sigma": float(np.median(np.abs(np.log(pooled / np.median(pooled)))) * 1.4826),
                     "capture_medians_spread": float(np.std([v["ratio_median"] for k, v in out.items() if k != "pooled"]))}
    print("pooled", out["pooled"])
    res = ROOT / "benchmark/results"
    res.mkdir(parents=True, exist_ok=True)
    (res / "metric_depth_calibration.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24)
