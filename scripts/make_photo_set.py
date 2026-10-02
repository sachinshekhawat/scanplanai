"""Derive a photo-tier benchmark input (per-room folders of stills) from a Stray capture.

Disclosure: these stills are frames of the capture's rgb.mp4 (1920x1440), re-encoded as
JPEG without EXIF, so the photo pipeline sees exactly what a photo-only user would provide
(no depth, no poses, no intrinsics). Room membership and the choice of stills use the LiDAR
result (camera inside the room polygon), which emulates a person standing in each room;
nothing from LiDAR is passed to the photo pipeline.

Selection mimics the protocol: from one standpoint cluster, 6 stills turning in place with
~50-70 deg between headings (overlap ~1/3 at a 69 deg HFOV), preferring views across the
room (large median depth).

    python scripts/make_photo_set.py data/raw/single_scan_with_ceiling data/photo_sets/with_ceiling [n=6]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scan2plan.lidar.pipeline import run_lidar  # noqa: E402


def main(src, dst, n=6):
    res = run_lidar(src, drift=True, log=lambda m: None)
    cap = res.cap
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    plan_xy = cap.positions[:, [0, 2]] @ res.R2.T
    yaw = np.arctan2(cap.rotations[:, 0, 2], cap.rotations[:, 2, 2])
    vc = cv2.VideoCapture(str(cap.root / "rgb.mp4"))
    rooms = sorted(res.seg.polygons.items(), key=lambda kv: -kv[1].area)
    order = []
    for lbl, poly in rooms:
        inside = np.nonzero(shapely.contains_xy(poly.buffer(-0.2), plan_xy[:, 0], plan_xy[:, 1]))[0]
        if len(inside) < 30 or poly.area < 1.5:
            continue
        order.append((int(np.median(inside)), lbl, poly, inside))
    order.sort()  # walking order = when the camera was in the room
    manifest = []
    for k, (_, lbl, poly, inside) in enumerate(order):
        # one visit: the longest contiguous run of frames inside the room (photos are taken in
        # one go), then stills spread in heading (>= 40 deg apart), largest median depth first
        runs = np.split(inside, np.nonzero(np.diff(inside) > 15)[0] + 1)
        cand = max(runs, key=len)
        med = np.array([np.median(cap.depth(i)[cap.confidence(i) == 2]) if (cap.confidence(i) == 2).any() else 0
                        for i in cand])
        sel = []
        for sep in (40, 25, 12):
            for i in cand[np.argsort(-med)]:
                if len(sel) == n:
                    break
                if all(abs(np.angle(np.exp(1j * (yaw[i] - yaw[j])))) > np.deg2rad(sep) for j in sel):
                    sel.append(int(i))
            if len(sel) >= max(2, n - 2):
                break
        sel.sort()
        folder = dst / f"{k + 1:02d}_room{lbl}"
        folder.mkdir(exist_ok=True)
        for i in sel:
            vc.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = vc.read()
            if ok:
                # Stray stores frames sideways; a phone stores photos upright (EXIF-applied)
                f = cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(str(folder / f"IMG_{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
        manifest.append(f"{folder.name}: frames {sel}, lidar room {lbl} area {poly.area:.2f} m2")
        print(manifest[-1])
    (dst / "MANIFEST.txt").write_text("derived from " + str(src) + "\n" + "\n".join(manifest) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
