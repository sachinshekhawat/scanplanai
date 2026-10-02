"""Bridge consecutive SfM fragments with Depth Anything 3 multi-view poses on a short window.

SfM breaks where frames lack texture or the camera turns quickly; each fragment is accurate
but lives in its own similarity frame. DA3 relative poses are reliable over short windows
(~sub-degree on consecutive frames in our tests) even on blank walls, so for each gap we run
DA3 on [last K frames of A] + [gap frames] + [first K frames of B] and use it as the
connecting rigid transform:

    T_world<-B = T_world<-A_anchor  *  T_DA3(A_anchor -> B_anchor)  *  T_B_anchor^-1

Rotation: chordal mean over anchor pairs. Translation: DA3 translation scaled to metres by the
median depth ratio (metric depth / DA3 depth) on the anchor frames.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

K_ANCHOR = 4
MAX_WINDOW = 28


def _so3_mean(Rs: list[np.ndarray]) -> np.ndarray:
    M = np.sum(Rs, axis=0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def bridge_windows(frags_names: list[list[str]], all_names: list[str]) -> list[list[str]]:
    """Frame-name windows spanning each gap between consecutive fragments."""
    idx = {n: i for i, n in enumerate(all_names)}
    wins = []
    for a, b in zip(frags_names[:-1], frags_names[1:]):
        ia, ib = idx[a[-1]], idx[b[0]]
        mid = all_names[ia + 1:ib]
        room = MAX_WINDOW - 2 * K_ANCHOR
        if len(mid) > room:
            mid = [mid[i] for i in np.linspace(0, len(mid) - 1, room).astype(int)]
        wins.append(a[-K_ANCHOR:] + mid + b[:K_ANCHOR])
    return wins


def run_da3_windows(wins: list[list[str]], frames_dir: Path, work: Path, log=print) -> list[dict]:
    """All gap windows in ONE DA3 child process (one model load)."""
    import json

    work.mkdir(parents=True, exist_ok=True)
    jobs = []
    for k, w in enumerate(wins):
        f = work / f"bridge_{k:02d}.npz"
        if not f.exists():
            lst = work / f"bridge_{k:02d}.txt"
            lst.write_text("\n".join(str((frames_dir / f"{n}.jpg").resolve()) for n in w))
            jobs.append([str(lst), str(f)])
    if jobs:
        (work / "jobs.json").write_text(json.dumps(jobs))
        p = subprocess.run([sys.executable, "-m", "scan2plan.learned.da3_worker", "multi", str(work / "jobs.json")],
                           capture_output=True, text=True)
        for line in p.stdout.splitlines():
            if line.startswith("[da3]"):
                log("[video] " + line)
    out = []
    for k in range(len(wins)):
        f = work / f"bridge_{k:02d}.npz"
        if not f.exists():
            out.append(None)
            continue
        z = np.load(f)
        out.append({"names": [Path(n).stem for n in z["names"]], "E": z["extrinsics"], "depth": z["depth"]})
    return out


def bridge_transform(da3: dict, A, B, metric_depth: dict[str, np.ndarray], orient: int = 0):
    """Return (R, t) mapping fragment B's world frame into fragment A's world frame (metric),
    plus a quality dict. A and B are PseudoCaptures (metric, own frames)."""
    import cv2

    names = da3["names"]
    E = da3["E"]
    from scan2plan.video.pipeline import ROT, UNROT

    M = ROT[orient][1]  # p_upright = M p_stored  =>  R_wc(stored) = R_wc(upright) @ M
    Rwc_d = np.array([e[:, :3].T @ M for e in E])  # DA3 cam->world, stored-camera convention
    C_d = np.array([-e[:, :3].T @ e[:, 3] for e in E])
    ia = {n: i for i, n in enumerate(A.names)}
    ib = {n: i for i, n in enumerate(B.names)}
    a_anc = [j for j, n in enumerate(names) if n in ia]
    b_anc = [j for j, n in enumerate(names) if n in ib]
    if not a_anc or not b_anc:
        return None, {"ok": False, "reason": "no anchors"}
    # metres per DA3 unit, from depth ratios on all anchor frames
    ratios = []
    for j in a_anc + b_anc:
        dm = metric_depth[names[j]]
        dj = da3["depth"][j].astype(np.float32)
        if orient:
            dj = cv2.rotate(dj, UNROT[orient])
        dd = cv2.resize(dj, (dm.shape[1], dm.shape[0]))
        m = (dd > 1e-3) & (dm > 0.2)
        if m.sum() > 100:
            ratios.append(np.median(dm[m] / dd[m]))
    if not ratios:
        return None, {"ok": False, "reason": "no depth overlap"}
    s = float(np.median(ratios))
    # rotation DA3-world -> A-world from A anchors; DA3-world -> B-world from B anchors
    R_A = _so3_mean([A.rotations[ia[names[j]]] @ Rwc_d[j].T for j in a_anc])
    R_B = _so3_mean([B.rotations[ib[names[j]]] @ Rwc_d[j].T for j in b_anc])
    # consistency of the anchor rotations (deg): a quality signal
    spread_a = max(np.degrees(np.arccos(np.clip((np.trace(R_A.T @ A.rotations[ia[names[j]]] @ Rwc_d[j].T) - 1) / 2, -1, 1)))
                   for j in a_anc)
    spread_b = max(np.degrees(np.arccos(np.clip((np.trace(R_B.T @ B.rotations[ib[names[j]]] @ Rwc_d[j].T) - 1) / 2, -1, 1)))
                   for j in b_anc)
    # translation: DA3 centres (scaled) -> A frame through anchor centroids
    cA_d = C_d[a_anc].mean(0) * s
    cA = np.mean([A.positions[ia[names[j]]] for j in a_anc], 0)
    cB_d = C_d[b_anc].mean(0) * s
    cB = np.mean([B.positions[ib[names[j]]] for j in b_anc], 0)
    # B-world -> DA3-world: R_B^T ; DA3-world -> A-world: R_A
    R = R_A @ R_B.T
    # a point x_B in B-world: x_d = R_B^T (x_B - cB) + cB_d ; x_A = R_A (x_d - cA_d) + cA
    t = R_A @ (cB_d - cA_d) + cA - R @ cB
    return (R, t), {"ok": True, "scale_m_per_unit": s, "anchor_rot_spread_deg": float(spread_a), "anchor_rot_spread_b_deg": float(spread_b),
                    "gap_translation_m": float(np.linalg.norm(R_A @ (cB_d - cA_d)))}
