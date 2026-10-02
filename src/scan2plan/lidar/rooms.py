"""Room segmentation, openings and rectilinear room polygons in the Manhattan plan frame.

All geometry is 2D in the "plan frame": world (x, z) rotated by the dominant wall yaw so
walls run along the axes.

 1. Free space: plan-view ray carving (cells seen through by >= 2 keyframes) minus
    structural walls (vertical evidence over most of the wall band height, which rejects
    furniture fronts).
 2. Wall faces: peaks of the 1D histogram of vertical points with axis-aligned normals,
    one line per (axis, normal sign, offset), refined to the median of inliers (sub-cm),
    split into runs along the line using the "door band" (below door headers).
 3. Openings: a gap between two collinear runs of the same face, 0.45-1.6 m wide, with
    carved free space through it. Closing those gaps in the wall image seals rooms.
 4. Rooms: components of free space with doors sealed, that the camera walked through.
 5. Polygon: contour of the room mask, rectilinearised, each edge snapped to the wall
    face that bounds it (face normal pointing into the room).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon

RES = 0.02
DOOR_MIN, DOOR_MAX = 0.45, 1.6


@dataclass
class Grid:
    x0: float
    y0: float
    w: int
    h: int
    res: float = RES

    def idx(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.floor((xy[:, 0] - self.x0) / self.res).astype(int)
        iy = np.floor((xy[:, 1] - self.y0) / self.res).astype(int)
        return np.clip(ix, 0, self.w - 1), np.clip(iy, 0, self.h - 1)

    def count(self, xy: np.ndarray) -> np.ndarray:
        ix, iy = self.idx(xy)
        img = np.zeros((self.h, self.w))
        np.add.at(img, (iy, ix), 1)
        return img

    def to_px(self, xy) -> np.ndarray:
        xy = np.atleast_2d(xy)
        return np.stack([(xy[:, 0] - self.x0) / self.res, (xy[:, 1] - self.y0) / self.res], 1)

    def to_m(self, px) -> np.ndarray:
        px = np.atleast_2d(px).astype(float)
        return np.stack([self.x0 + px[:, 0] * self.res, self.y0 + px[:, 1] * self.res], 1)


@dataclass
class WallRun:
    """One contiguous run of a wall face. axis=0: face is the line x=c; axis=1: y=c.
    sign: direction (+1/-1 along `axis`) the face normal points, i.e. where the room is."""
    axis: int
    sign: int
    c: float
    sigma: float
    lo: float
    hi: float
    support: int
    line_id: int = -1


@dataclass
class Opening:
    axis: int
    c: float  # wall face coordinate
    lo: float  # jamb positions along the wall
    hi: float
    kind: str = "door"  # door | open_passage | window
    rooms: tuple = ()  # room labels it connects
    jamb_refined: tuple = (False, False)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def endpoints(self) -> np.ndarray:
        if self.axis == 0:
            return np.array([[self.c, self.lo], [self.c, self.hi]])
        return np.array([[self.lo, self.c], [self.hi, self.c]])


@dataclass
class RoomSeg:
    grid: Grid
    wall_img: np.ndarray
    sealed: np.ndarray
    free: np.ndarray
    labels: np.ndarray
    runs: list[WallRun]
    openings: list[Opening]
    polygons: dict[int, Polygon] = field(default_factory=dict)
    edges: dict[int, list] = field(default_factory=dict)  # per room: list of edge dicts


def disk(r_px: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))


def make_grid(xy: np.ndarray, pad: float = 0.5) -> Grid:
    lo = xy.min(0) - pad
    hi = xy.max(0) + pad
    w, h = np.ceil((hi - lo) / RES).astype(int) + 1
    return Grid(float(lo[0]), float(lo[1]), int(w), int(h))


FIXED_BAND = False  # fix loop: band must not depend on whether this session saw the ceiling


def wall_band(floor_y: float, ceil_y: float | None) -> tuple[float, float]:
    if FIXED_BAND:
        return floor_y + 0.15, floor_y + 2.0
    top = (ceil_y - 0.25) if ceil_y is not None else floor_y + 2.2
    return floor_y + 0.15, top


def carve_free(grid: Grid, cam_xy: np.ndarray, fans: np.ndarray, R2: np.ndarray,
               shrink: float = 0.04, min_views: int = 2) -> np.ndarray:
    """Rasterise per-keyframe plan-view visibility fans; a cell is free if >= min_views see through it."""
    nb = fans.shape[1]
    bearing = (np.arange(nb) + 0.5) / nb * 2 * np.pi - np.pi
    dirs = np.stack([np.cos(bearing), np.sin(bearing)], 1) @ R2.T
    acc = np.zeros((grid.h, grid.w), np.uint16)
    img = np.zeros((grid.h, grid.w), np.uint8)
    for k in range(len(fans)):
        r = fans[k]
        ok = np.isfinite(r)
        if ok.sum() < 3:
            continue
        rr = np.where(ok, np.maximum(r - shrink, 0), 0.0)
        px = grid.to_px(cam_xy[k] + dirs * rr[:, None])
        c = grid.to_px(cam_xy[k])[0]
        idx = np.nonzero(ok)[0]
        tri = [np.array([c, px[a], px[b]]) for a, b in zip(idx[:-1], idx[1:]) if b - a <= 2]
        if not tri:
            continue
        img[:] = 0
        cv2.fillPoly(img, [np.round(t * 4).astype(np.int32) for t in tri], 1, shift=2)
        acc += img
    return acc >= min_views


def structural_walls(grid: Grid, vxy: np.ndarray, vy: np.ndarray, band: tuple[float, float],
                     slab: float = 0.1, min_cov: float = 0.6) -> np.ndarray:
    """Cells whose vertical evidence covers >= min_cov of the band's height slabs.
    Furniture fronts cover only the lower part and are rejected."""
    nsl = max(int((band[1] - band[0]) / slab), 1)
    inb = (vy > band[0]) & (vy < band[1])
    ix, iy = grid.idx(vxy[inb])
    k = np.clip(((vy[inb] - band[0]) / slab).astype(int), 0, nsl - 1)
    key = np.unique((iy.astype(np.int64) * grid.w + ix) * nsl + k)
    cov = np.bincount(key // nsl, minlength=grid.w * grid.h).reshape(grid.h, grid.w)
    covm = ndi.maximum_filter(cov, size=3)
    img = (covm >= min_cov * nsl).astype(np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, disk(2))


def extract_runs(vxy: np.ndarray, vn: np.ndarray, vy: np.ndarray, floor_y: float,
                 band_top: float, min_len: float = 0.12, bin_m: float = 0.01,
                 split_gap: float = 0.12) -> list[WallRun]:
    """Axis-aligned wall face runs. Offsets use the full wall band; extents along the face
    use the door band (floor+0.3 .. floor+1.85) so door headers do not bridge doorways."""
    runs: list[WallRun] = []
    full = (vy > floor_y + 0.15) & (vy < band_top)
    door_band = (vy > floor_y + 0.3) & (vy < floor_y + 1.85)
    min_support = 150
    line_id = 0
    for axis in (0, 1):
        for sign in (1, -1):
            sel = full & (vn[:, axis] * sign > 0.9)
            p, pd = vxy[sel], door_band[sel]
            if len(p) < min_support:
                continue
            c = p[:, axis]
            hist, edges = np.histogram(c, bins=np.arange(c.min() - bin_m, c.max() + 2 * bin_m, bin_m))
            sm = np.convolve(hist, [0.25, 0.5, 0.25], mode="same")
            mx = ndi.maximum_filter1d(sm, 7)
            for pk in np.where((sm == mx) & (sm >= min_support / 3))[0]:
                centre = edges[pk] + bin_m / 2
                m = np.abs(c - centre) < 0.03
                if m.sum() < min_support:
                    continue
                med = float(np.median(c[m]))
                m = np.abs(c - med) < 0.02
                med = float(np.median(c[m]))
                sig = float(np.median(np.abs(c[m] - med)) * 1.4826)
                # extents from door-band points; occupancy along the face in 2 cm cells
                a = p[m & pd, 1 - axis]
                if len(a) < 30:
                    continue
                lo0 = a.min()
                cells = np.bincount(((a - lo0) / 0.02).astype(int))
                occ = cells >= 3  # >= 3 voxels per 2 cm cell across ~1.5 m of height
                occ = ndi.binary_closing(occ, np.ones(3)) | occ
                lab, n = ndi.label(occ)
                for k in range(1, n + 1):
                    idx = np.nonzero(lab == k)[0]
                    s, e = lo0 + idx[0] * 0.02, lo0 + (idx[-1] + 1) * 0.02
                    if e - s < min_len:
                        continue
                    # refine jamb ends with in-run point quantiles (robust to stray voxels)
                    aa = a[(a >= s - 0.01) & (a <= e + 0.01)]
                    s2, e2 = np.quantile(aa, [0.002, 0.998])
                    runs.append(WallRun(axis, sign, med, sig, float(s2), float(e2), int(len(aa)), line_id))
                line_id += 1
    # merge runs separated by tiny gaps (< split_gap) on the same line
    runs.sort(key=lambda r: (r.line_id, r.lo))
    merged: list[WallRun] = []
    for r in runs:
        if merged and merged[-1].line_id == r.line_id and r.lo - merged[-1].hi < split_gap:
            q = merged[-1]
            q.hi = max(q.hi, r.hi)
            q.support += r.support
        else:
            merged.append(r)
    return merged


def _face_xy(axis: int, c: float, t: float) -> np.ndarray:
    return np.array([c, t]) if axis == 0 else np.array([t, c])


def axis_close(wall_img: np.ndarray, max_gap: float, min_run: float = 0.06) -> np.ndarray:
    """Bridge doorway gaps in straight walls only. Wall pixels are split by orientation
    (opening with a min_run line kernel keeps only runs along that axis), and each set is
    closed along its own axis. This seals doorways but never fills a corridor: a corridor's
    two side walls are parallel to the corridor, not collinear across it."""
    L = int(round(max_gap / RES)) | 1
    m = int(round(min_run / RES)) | 1
    horiz = cv2.morphologyEx(wall_img, cv2.MORPH_OPEN, np.ones((1, m), np.uint8))
    vert = cv2.morphologyEx(wall_img, cv2.MORPH_OPEN, np.ones((m, 1), np.uint8))
    h = cv2.morphologyEx(horiz, cv2.MORPH_CLOSE, np.ones((1, L), np.uint8))
    v = cv2.morphologyEx(vert, cv2.MORPH_CLOSE, np.ones((L, 1), np.uint8))
    return wall_img | h | v


def _jamb_refine(o: "Opening", runs: list[WallRun], tol: float = 0.1) -> "Opening":
    """Snap opening ends to wall-face run ends (jambs) near the opening line. Parallel faces
    within 0.3 m of the opening line (either face of the wall) and perpendicular faces that
    cross it (door beside a corner) both count."""
    best_lo = best_hi = None
    for r in runs:
        if r.axis == o.axis and abs(r.c - o.c) <= 0.3:
            cand_lo, cand_hi = r.hi, r.lo
        elif r.axis != o.axis and r.lo - 0.05 <= o.c <= r.hi + 0.05:
            cand_lo = cand_hi = r.c
        else:
            continue
        if abs(cand_lo - o.lo) < tol and (best_lo is None or abs(cand_lo - o.lo) < abs(best_lo - o.lo)):
            best_lo = cand_lo
        if abs(cand_hi - o.hi) < tol and (best_hi is None or abs(cand_hi - o.hi) < abs(best_hi - o.hi)):
            best_hi = cand_hi
    lo = best_lo if best_lo is not None else o.lo
    hi = best_hi if best_hi is not None else o.hi
    return Opening(o.axis, o.c, lo, hi, o.kind, o.rooms, (best_lo is not None, best_hi is not None))


def find_openings(wall_img: np.ndarray, sealed: np.ndarray, labels: np.ndarray, free: np.ndarray,
                  grid: Grid, runs: list[WallRun], ring_px: int = 14) -> list[Opening]:
    """Openings = closure patches (sealed but not structural wall) lying on free space whose
    neighbourhood touches >= 2 rooms. Width: patch extent along the wall, then snapped to
    jambs from the wall-face runs (sub-cm when both jambs are found)."""
    patch = (sealed > 0) & (wall_img == 0) & free
    lab, _ = ndi.label(patch, structure=np.ones((3, 3)))
    out: list[Opening] = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        if m.sum() < 8:
            continue
        y0, y1 = max(sl[0].start - ring_px, 0), min(sl[0].stop + ring_px, grid.h)
        x0, x1 = max(sl[1].start - ring_px, 0), min(sl[1].stop + ring_px, grid.w)
        big = np.zeros((y1 - y0, x1 - x0), np.uint8)
        big[sl[0].start - y0:sl[0].stop - y0, sl[1].start - x0:sl[1].stop - x0] = m
        ring = cv2.dilate(big, disk(ring_px)).astype(bool)
        lv = labels[y0:y1, x0:x1][ring]
        cnt = np.bincount(lv[lv > 0])
        rl = [int(r) for r in np.argsort(cnt)[::-1] if r > 0 and cnt[r] >= 10][:2]
        if len(rl) < 2:
            continue
        ys, xs = np.nonzero(m)
        ext_x, ext_y = np.ptp(xs) + 1, np.ptp(ys) + 1
        along = 0 if ext_x >= ext_y else 1
        axis = 1 - along
        pts = grid.to_m(np.stack([xs + sl[1].start, ys + sl[0].start], 1) + 0.5)
        c = float(np.median(pts[:, axis]))
        lo = float(pts[:, along].min() - grid.res / 2)
        hi = float(pts[:, along].max() + grid.res / 2)
        o = _jamb_refine(Opening(axis, c, lo, hi, "door", tuple(sorted(rl))), runs)
        if not (DOOR_MIN - 0.1 <= o.width <= DOOR_MAX + 0.1):
            continue
        o.kind = "door" if o.width <= 1.1 else "open_passage"
        out.append(o)
    # one opening per (room pair, location): merge fragments of the same doorway
    out.sort(key=lambda o: -o.width)
    dedup: list[Opening] = []
    for o in out:
        if any(d.rooms == o.rooms and d.axis == o.axis and abs(d.c - o.c) < 0.3
               and min(d.hi, o.hi) - max(d.lo, o.lo) > -0.05 for d in dedup):
            continue
        dedup.append(o)
    return dedup


def segment_rooms(vxy: np.ndarray, vn: np.ndarray, vy: np.ndarray, cam_xy: np.ndarray,
                  fans: np.ndarray, R2: np.ndarray, floor_y: float, ceil_y: float | None,
                  min_room_area: float = 1.0, min_cam_px: int = 15) -> RoomSeg:
    band = wall_band(floor_y, ceil_y)
    grid = make_grid(np.concatenate([vxy[::10], cam_xy]))
    wall_img = structural_walls(grid, vxy, vy, band)
    carved = carve_free(grid, cam_xy, fans, R2)
    ci, cj = grid.idx(cam_xy)
    cam = np.zeros((grid.h, grid.w), np.uint8)
    cam[cj, ci] = 1
    free = (carved | (cv2.dilate(cam, disk(5)) > 0)) & (cv2.dilate(wall_img, disk(1)) == 0)
    free = ndi.binary_opening(free, structure=disk(2))

    runs = extract_runs(vxy, vn, vy, floor_y, band[1])
    sealed = axis_close(wall_img, DOOR_MAX)
    rooms_free = free & (cv2.dilate(sealed, disk(1)) == 0)
    lab, n = ndi.label(rooms_free)
    labels = np.zeros_like(lab)
    k = 0
    cam_d = cv2.dilate(cam, disk(3)) > 0
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() * RES**2 < min_room_area or (m & cam_d).sum() < min_cam_px:
            continue  # too small, or seen through a doorway but never entered
        k += 1
        labels[m] = k
    openings = find_openings(wall_img, sealed, labels, free, grid, runs)
    seg = RoomSeg(grid, wall_img, sealed, free, labels, runs, openings)
    for k in range(1, labels.max() + 1):
        poly, edges = room_polygon(labels == k, grid, runs)
        if not poly.is_empty:
            seg.polygons[k] = poly
            seg.edges[k] = edges
    return seg


# Fix loop (docs/fix_loop.md): edges bounded only by free space are pushed outward to the
# nearest wall face; short jogs between edges that snap to the same face are collapsed.
SNAP_OUTWARD = False  # tried in fix loop: no repeatability gain (see docs/fix_loop.md)
OUT_SNAP_MAX = 0.6
JOG_MAX = 0.35


AMBIGUITY = True  # fix loop: per-edge face-ambiguity term in the error model
AMB_REACH = 0.6


def _edge_ambiguity(e: dict, runs: list[WallRun]) -> float:
    """1-sigma for 'which face bounds this edge': half the spread of alternative compatible
    faces (same axis and normal, overlapping the edge) within AMB_REACH. Inner faces such as
    wardrobe fronts, radiators and partition stubs make the room boundary depend on what a
    walk happened to observe. Unsnapped edges get at least half the distance to the nearest
    face, or AMB_REACH/2 if none exists."""
    alts = []
    for r in runs:
        if r.axis != e["axis"] or r.sign != e["inward"]:
            continue
        ov = min(e["hi"], r.hi) - max(e["lo"], r.lo)
        if ov > 0.2 * (e["hi"] - e["lo"]) and abs(r.c - e["c"]) < AMB_REACH:
            alts.append(r.c)
    if not e["snapped"]:
        return 0.5 * (min(abs(a - e["c"]) for a in alts) if alts else AMB_REACH)
    spread = max(alts) - min(alts) if len(alts) > 1 else 0.0
    return 0.5 * spread


def _snap_outward(edges: list[dict], runs: list[WallRun]) -> list[dict]:
    for e in edges:
        # every edge (snapped or not) moves to the OUTERMOST compatible face within reach:
        # inner faces (wardrobe fronts, radiators, shelving) are what one session sees and the
        # other does not; the structural wall behind them is common to both.
        ax, inward = e["axis"], e["inward"]
        # free space stops at or before the wall: the true face lies on the outward side,
        # i.e. at c' with (c' - c) * inward <= 0, and the face's normal must point inward
        best, bd = None, -1.0
        for r in runs:
            if r.axis != ax or r.sign != inward:
                continue
            d = (e["c"] - r.c) * inward
            ov = min(e["hi"], r.hi) - max(e["lo"], r.lo)
            reach = OUT_SNAP_MAX if not e["snapped"] else 0.12
            if -0.03 <= d < reach and ov > 0.3 * (e["hi"] - e["lo"]) and (best is None or d > bd):
                best, bd = r, d
        if best is not None:
            e.update(c=best.c, snapped=True, sigma=best.sigma, support=best.support)
    # collapse jogs: A(axis k, c1) - J(short, other axis) - B(axis k, c2) with |c1-c2| small
    # after snapping both to the same face -> merge into one edge
    changed = True
    while changed and len(edges) > 4:
        changed = False
        n = len(edges)
        for i in range(n):
            a, j, b = edges[i], edges[(i + 1) % n], edges[(i + 2) % n]
            if a["axis"] == b["axis"] and j["axis"] != a["axis"] and abs(a["c"] - b["c"]) < 0.02 \
                    and (j["hi"] - j["lo"]) < JOG_MAX:
                a["lo"], a["hi"] = min(a["lo"], b["lo"]), max(a["hi"], b["hi"])
                for idx in sorted({(i + 1) % n, (i + 2) % n}, reverse=True):
                    del edges[idx]
                changed = True
                break
    return edges


def room_polygon(mask: np.ndarray, grid: Grid, runs: list[WallRun], snap: float = 0.15,
                 min_edge: float = 0.08) -> tuple[Polygon, list[dict]]:
    """Rectilinear polygon of a room mask with edges snapped to bounding wall faces.
    Returns the polygon and per-edge metadata (snapped run, sigma) for uncertainty."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, disk(4))
    m = ndi.binary_fill_holes(m).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return Polygon(), []
    cnt = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    approx = cv2.approxPolyDP(cnt.astype(np.float32), 3.0, True)[:, 0, :].astype(float)
    pts = grid.to_m(approx + 0.5)
    n = len(pts)
    # classify each edge as along-x (horizontal, y=const => axis 1) or along-y (axis 0)
    segs = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        d = b - a
        L = np.hypot(*d)
        if L < 1e-6:
            continue
        axis = 1 if abs(d[0]) >= abs(d[1]) else 0
        segs.append({"axis": axis, "a": a, "b": b, "len": L})
    # merge consecutive segments with the same axis
    merged = []
    for s in segs:
        if merged and merged[-1]["axis"] == s["axis"]:
            q = merged[-1]
            q["b"] = s["b"]
            q["len"] += s["len"]
        else:
            merged.append(dict(s))
    if len(merged) > 1 and merged[0]["axis"] == merged[-1]["axis"]:
        merged[0]["a"] = merged[-1]["a"]
        merged[0]["len"] += merged[-1]["len"]
        merged.pop()
    # drop tiny jogs (< min_edge) by absorbing into neighbours, repeat until stable
    changed = True
    while changed and len(merged) > 4:
        changed = False
        for i, s in enumerate(merged):
            if s["len"] < min_edge:
                prev, nxt = merged[i - 1], merged[(i + 1) % len(merged)]
                prev["b"] = nxt["b"]
                prev["len"] += s["len"] + nxt["len"]
                j = (i + 1) % len(merged)
                for idx in sorted({i, j}, reverse=True):
                    del merged[idx]
                changed = True
                break
    # interior side: polygon orientation (shapely signed area)
    ccw = Polygon(pts).exterior.is_ccw
    edges = []
    for s in merged:
        ax = s["axis"]
        coords = np.array([s["a"], s["b"]])
        c_raw = float(np.median(coords[:, ax]))
        lo, hi = sorted(coords[:, 1 - ax])
        d = s["b"] - s["a"]
        # inward normal: left of travel direction for CCW
        nrm = np.array([-d[1], d[0]]) if ccw else np.array([d[1], -d[0]])
        inward = int(np.sign(nrm[ax])) or 1
        best, bd = None, snap
        for r in runs:
            if r.axis != ax or r.sign != inward:
                continue
            ov = min(hi, r.hi) - max(lo, r.lo)
            if ov < 0.3 * min(hi - lo, r.hi - r.lo) and ov < 0.25:
                continue
            dist = abs(r.c - c_raw)
            if dist < bd:
                best, bd = r, dist
        edges.append({"axis": ax, "c": best.c if best else c_raw, "lo": lo, "hi": hi,
                      "snapped": best is not None, "sigma": best.sigma if best else 0.03,
                      "support": best.support if best else 0, "inward": inward})
    if SNAP_OUTWARD:
        edges = _snap_outward(edges, runs)
    if AMBIGUITY:
        for e in edges:
            e["amb"] = _edge_ambiguity(e, runs)
    # corners = intersections of consecutive (alternating-axis) edges
    corners = []
    for i in range(len(edges)):
        e, f = edges[i - 1], edges[i]
        if e["axis"] == f["axis"]:
            continue
        x = e["c"] if e["axis"] == 0 else f["c"]
        y = e["c"] if e["axis"] == 1 else f["c"]
        corners.append((x, y))
    if len(corners) < 3:
        return Polygon(), []
    poly = Polygon(corners)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.geom_type != "Polygon":
        parts = [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
        if not parts:
            return Polygon(), []
        poly = max(parts, key=lambda g: g.area)
    # recompute edge extents from final corners
    final = []
    cs_ = list(poly.exterior.coords)[:-1] if poly.geom_type == "Polygon" else []
    for i in range(len(cs_)):
        a, b = np.array(cs_[i]), np.array(cs_[(i + 1) % len(cs_)])
        ax = 0 if abs(b[0] - a[0]) < abs(b[1] - a[1]) else 1
        c = (a[ax] + b[ax]) / 2
        match = min(edges, key=lambda e: (e["axis"] != ax, abs(e["c"] - c)))
        final.append({**match, "a": a.tolist(), "b": b.tolist(), "length": float(np.hypot(*(b - a)))})
    return poly, final
