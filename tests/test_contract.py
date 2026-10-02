"""Unit tests for the output contract and geometry primitives (no sample data needed)."""
import json

import numpy as np
import pytest
from shapely.geometry import box

from scan2plan.lidar.rooms import RES, WallRun, axis_close, extract_runs, make_grid, room_polygon
from scan2plan.measure import Z95, Measurement, length_sigma, unobserved
from scan2plan.plan import Plan, PlanEdge, PlanOpening, PlanRoom, to_document
from scan2plan.schema import validate


def _rect_room(x0, y0, x1, y1, sigma=0.005):
    poly = box(x0, y0, x1, y1)
    c = list(poly.exterior.coords)[:-1]
    edges = [PlanEdge(np.array(c[i]), np.array(c[(i + 1) % 4]), sigma, True) for i in range(4)]
    return PlanRoom(poly, edges, Measurement(2.5, 0.01, "test"))


def _plan():
    r1, r2 = _rect_room(0, 0, 4, 3), _rect_room(4.1, 0, 7, 3)
    door = PlanOpening(np.array([4.05, 1.0]), np.array([4.05, 1.9]), 0.008, "door", (0, 1), "test")
    return Plan("lidar", [r1, r2], [door], {"name": "unit", "tier": "lidar"})


def test_measurement_interval_contains_value_and_is_95pct():
    m = Measurement(3.0, 0.01, "x").to_json()
    assert m["ci95"][0] < 3.0 < m["ci95"][1]
    assert np.isclose(m["ci95"][1] - m["ci95"][0], 2 * Z95 * 0.01, atol=1e-4)
    assert unobserved(2.6, 0.3, "prior").to_json()["observed"] is False


def test_length_sigma_grows_with_length_and_face_noise():
    assert length_sigma(5, 0.005, 0.005) > length_sigma(1, 0.005, 0.005)
    assert length_sigma(3, 0.02, 0.005) > length_sigma(3, 0.005, 0.005)


def test_document_validates_and_has_adjacency():
    doc = to_document(_plan())
    validate(doc)
    json.dumps(doc)
    assert doc["property"]["n_rooms"] == 2
    adj = doc["property"]["adjacency"]
    assert adj and adj[0]["rooms"] == ["R1", "R2"] and adj[0]["via"] == ["O1"]
    assert not doc["property"]["overlaps"]
    walls = {w["id"]: w["length"]["value"] for r in doc["rooms"] for w in r["walls"]}
    assert sorted(round(v, 3) for v in walls.values()) == [2.9, 2.9, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0]


def test_validator_rejects_dangling_reference():
    doc = to_document(_plan())
    doc["scope"] = [{"id": "S1", "surface": "R9-FLOOR", "action": "x",
                     "quantity": Measurement(1, 0.1, "t").to_json(), "unit": "m2"}]
    with pytest.raises(ValueError):
        validate(doc)


def test_axis_close_seals_doorway_but_not_corridor():
    img = np.zeros((200, 200), np.uint8)
    img[100, 20:80] = 1
    img[100, 125:180] = 1  # 0.9 m doorway between the two wall pieces
    img[20:60, 40] = 1
    img[20:60, 90] = 1  # corridor: two parallel walls 1.0 m apart must stay open
    out = axis_close(img, 1.6)
    assert out[100, 80:125].all()
    assert not out[40, 41:90].any()


def test_wall_runs_and_polygon_recover_rectangle():
    rng = np.random.default_rng(0)
    pts, nrm = [], []
    W, H = 4.0, 3.0
    for (axis, c, sign, L) in ((0, 0.0, 1, H), (0, W, -1, H), (1, 0.0, 1, W), (1, H, -1, W)):
        t = rng.uniform(0, L, 6000)
        q = np.zeros((len(t), 2))
        q[:, axis] = c + rng.normal(0, 0.003, len(t))
        q[:, 1 - axis] = t
        n = np.zeros((len(t), 2))
        n[:, axis] = sign
        pts.append(q)
        nrm.append(n)
    vxy, vn = np.concatenate(pts), np.concatenate(nrm)
    vy = rng.uniform(0.2, 2.0, len(vxy))
    runs = extract_runs(vxy, vn, vy, 0.0, 2.2)
    assert {(r.axis, r.sign) for r in runs} == {(0, 1), (0, -1), (1, 1), (1, -1)}
    grid = make_grid(vxy)
    mask = np.zeros((grid.h, grid.w), bool)
    ix, iy = grid.idx(np.array([[0.1, 0.1], [W - 0.1, H - 0.1]]))
    mask[iy[0]:iy[1], ix[0]:ix[1]] = True
    poly, edges = room_polygon(mask, grid, runs)
    assert abs(poly.area - W * H) < 0.05
    assert all(e["snapped"] for e in edges)
