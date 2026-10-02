"""Child-process LoFTR (indoor) matching: python -m scan2plan.learned.loftr_worker <pairs.json> <out.npz>

pairs.json: [[path_a, path_b, key], ...]. Output: key -> (M,5) float32 [xa, ya, xb, yb, conf]
in the input images' pixel coordinates. Weights (kornia LoFTR 'indoor_new', ScanNet-trained)
are fetched once into the torch hub cache.
"""
import json
import sys

import cv2
import numpy as np


def main(pairs_json, out):
    import kornia.feature as KF
    import torch

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    m = KF.LoFTR(pretrained="indoor_new").eval().to(dev)
    cache = {}

    def load(p):
        if p not in cache:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            H, W = g.shape
            tw, th = (640, 480) if W >= H else (480, 640)
            cache[p] = (torch.from_numpy(cv2.resize(g, (tw, th)))[None, None].float().div(255).to(dev), W / tw, H / th)
        return cache[p]

    res = {}
    for a, b, key in json.load(open(pairs_json)):
        ia, sxa, sya = load(a)
        ib, sxb, syb = load(b)
        with torch.no_grad():
            o = m({"image0": ia, "image1": ib})
        k0 = o["keypoints0"].cpu().numpy() * [sxa, sya]
        k1 = o["keypoints1"].cpu().numpy() * [sxb, syb]
        c = o["confidence"].cpu().numpy()[:, None]
        res[key] = np.concatenate([k0, k1, c], 1).astype(np.float32)
    np.savez(out, **res)
    print(f"[loftr] {len(res)} pairs ({dev})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
