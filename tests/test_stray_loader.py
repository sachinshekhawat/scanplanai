from pathlib import Path

import numpy as np
import pytest

from scan2plan.io.stray import load_stray

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
CAPTURES = sorted(p.name for p in RAW.iterdir()) if RAW.exists() else []


@pytest.mark.skipif(not CAPTURES, reason="raw data not unpacked; run scripts/fetch_data.sh")
@pytest.mark.parametrize("name", CAPTURES)
def test_floor_is_dominant_low_plane(name):
    cap = load_stray(RAW / name)
    assert cap.n_frames > 100
    assert np.allclose(np.linalg.det(cap.rotations[:10]), 1.0, atol=1e-5)
    pts = np.concatenate([cap.backproject(i, stride=4) for i in range(0, cap.n_frames, 50)])
    y = pts[:, 1]
    hist, edges = np.histogram(y, bins=np.arange(y.min(), y.max(), 0.02))
    floor = edges[np.argmax(hist)]
    # Gravity-aligned +y-up world with OpenCV camera convention: floor is the
    # densest 2 cm slab and sits in the lowest few percent of points.
    assert floor < np.percentile(y, 10)
    # Ceiling captures split mass between floor and ceiling (~5%); the wrong
    # (ARKit -z) convention smears the floor to <4% even on the simplest room.
    assert hist.max() / hist.sum() > 0.045
