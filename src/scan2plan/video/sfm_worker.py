"""Child-process COLMAP SfM (pycolmap and torch cannot share a process on macOS: libomp clash).

    python -m scan2plan.video.sfm_worker <img_dir> <work>

SIFT tuned for low-texture interiors + sequential matching (quadratic overlap) + incremental
mapping. Every model with >= MIN_MODEL images is exported to <work>/sparse/export.npz as
plain arrays (one entry per fragment); the parent stitches fragments.

Tried and rejected (see report): LoFTR matches on weak consecutive pairs registered more
frames but produced geometrically inconsistent models (pairwise distance-ratio CV 1.9 vs
0.02 for SIFT-only fragments) and cost ~13 min per minute of video on an M1.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np

MAX_SIDE = 1024
MIN_MODEL = 8


def run(img_dir: Path, work: Path):
    import cv2
    import pycolmap

    db = work / "db.db"
    if db.exists():
        db.unlink()
    first = cv2.imread(str(sorted(img_dir.glob("*.jpg"))[0]))
    h, w = first.shape[:2]
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.camera_params = f"{0.83 * max(w, h)},{w / 2},{h / 2},0"  # iPhone 1x main-camera prior
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = MAX_SIDE
    ext.sift.max_num_features = 8192
    ext.sift.peak_threshold = 0.0015  # default 0.0067 finds ~400 kp/img on painted walls
    ext.sift.domain_size_pooling = True
    t = time.time()
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext)
    pair = pycolmap.SequentialPairingOptions()
    pair.overlap = 20
    pair.quadratic_overlap = True
    pycolmap.match_sequential(db, pairing_options=pair)
    print(f"[video] SIFT + sequential matching in {time.time() - t:.0f} s", flush=True)
    out = work / "sp"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    t = time.time()
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_model_size = MIN_MODEL
    recs = pycolmap.incremental_mapping(db, img_dir, out, opts)
    models = sorted((r for r in recs.values() if r.num_reg_images() >= MIN_MODEL),
                    key=lambda r: min(im.name for im in r.images.values()))
    n_img = len(list(img_dir.glob("*.jpg")))
    sizes = [r.num_reg_images() for r in models]
    print(f"[video] incremental SfM in {time.time() - t:.0f} s: {len(models)} fragments {sizes}, "
          f"{sum(sizes)}/{n_img} frames registered", flush=True)
    (work / "sparse").mkdir(exist_ok=True)
    np.savez(work / "sparse" / "export.npz", data=np.array([export(r) for r in models], dtype=object))


def export(rec) -> dict:
    imgs = list(rec.images.values())
    cam = rec.cameras[imgs[0].camera_id]
    pts = {pid: p.xyz for pid, p in rec.points3D.items()}
    data = {"wh": (int(cam.width), int(cam.height)), "params": np.array(cam.params, float), "images": []}
    for im in imgs:
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R, t = T.rotation.matrix(), np.array(T.translation)
        uv, z = [], []
        for p2 in im.points2D:
            if p2.has_point3D():
                zc = (R @ pts[p2.point3D_id] + t)[2]
                if zc > 0:
                    uv.append(p2.xy)
                    z.append(zc)
        data["images"].append({"name": im.name, "R": R, "t": t, "uv": np.array(uv).reshape(-1, 2),
                               "z": np.array(z)})
    return data


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
