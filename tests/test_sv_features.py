"""sv.dataset.features / sv.dataset.collections (torch-free): the feature derivations, the sentinel handling, the
vertex classes and the per-vertex track counts.

    PYTHONPATH=src python -m pytest tests/test_sv_features.py -q
"""

import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sv.dataset import collections as col  # noqa: E402
from sv.dataset import features as ft  # noqa: E402


def test_track_features_cover_their_branches_and_are_finite():
    rng = np.random.default_rng(0)
    n = 50
    table = {
        "track_pt": rng.uniform(0.8, 50, n), "track_eta": rng.uniform(-2.5, 2.5, n), "track_phi": rng.uniform(-np.pi, np.pi, n),
        "track_ip2dValue": rng.uniform(0, 1, n), "track_ip2dSignificance": rng.uniform(0, 50, n),
        "track_ip3dValue": rng.uniform(0, 5, n), "track_ip3dSignificance": rng.uniform(0, 500, n),
        "track_dz": rng.uniform(-10, 10, n), "track_dzSignificance": rng.uniform(-500, 500, n),
        "track_charge": rng.choice([-1, 1], n), "track_nHits": rng.integers(3, 30, n), "track_numberOfPixelHits": rng.integers(0, 8, n),
        "track_numberOfStripHits": rng.integers(0, 25, n), "track_chi2": rng.integers(0, 100, n), "track_qoverp": rng.uniform(-1, 1, n),
        "track_isHighPurity": rng.integers(0, 2, n).astype(bool), "track_passSeed": rng.integers(0, 2, n),
    }
    names = list(ft.TRACK_FEATURES)
    assert set(ft.branches_needed(ft.TRACK_FEATURES, names)) <= set(table)
    derived = ft.derive(ft.TRACK_FEATURES, names, table)
    assert set(derived) == set(names)
    for name, values in derived.items():
        assert values.shape == (n,) and np.isfinite(values).all(), name
    assert np.allclose(derived["pt"], np.log(table["track_pt"]))
    assert np.allclose(derived["sinphi"] ** 2 + derived["cosphi"] ** 2, 1)
    assert set(np.unique(derived["charge"])) <= {-1.0, 1.0}
    assert np.allclose(derived["highpurity"], table["track_isHighPurity"].astype(float))


def test_sentinels_map_to_zero_and_asinh_compresses():
    table = {"track_ip3dSignificance": np.array([-9999.0, 0.0, 1.0, 1e4]), "track_dz": np.array([-9999.0, -0.01, 0.0, 100.0])}
    sig = ft.derive(ft.TRACK_FEATURES, ["ip3dsig"], table)["ip3dsig"]
    assert sig[0] == 0 and sig[1] == 0 and abs(sig[2] - np.arcsinh(1.0)) < 1e-12 and sig[3] == pytest.approx(np.arcsinh(1e4))
    dz = ft.derive(ft.TRACK_FEATURES, ["dz"], table)["dz"]
    assert dz[0] == 0 and dz[1] == pytest.approx(np.arcsinh(-1.0)) and dz[2] == 0 and dz[3] == pytest.approx(np.arcsinh(100 / ft.IP_SCALE_CM))
    assert (dz[1] < 0) and (dz[3] > 0)  # sign preserved


def test_unknown_feature_is_refused():
    with pytest.raises(ValueError, match="unknown features"):
        ft.derive(ft.TRACK_FEATURES, ["pt", "nonsense"], {"track_pt": np.ones(2)})


def test_vertex_targets_are_measured_from_the_primary_vertex():
    table = {
        "GV_pt": np.array([20.0, 50.0]), "GV_eta": np.array([0.5, -1.0]), "GV_phi": np.array([0.0, np.pi / 2]),
        "GV_x": np.array([1.0, 0.1]), "GV_y": np.array([0.0, 0.2]), "GV_z": np.array([3.0, -1.0]),
        "PV_x": np.array([0.0, 0.1]), "PV_y": np.array([0.0, 0.2]), "PV_z": np.array([1.0, 1.0]),
    }
    d = ft.derive(ft.VERTEX_TARGETS, list(ft.VERTEX_TARGETS), table)
    assert np.allclose(d["lxy"], [1.0, 0.0]) and np.allclose(d["lz"], [2.0, -2.0]) and np.allclose(d["l3d"], [np.sqrt(5), 2.0])
    assert np.allclose(d["x"], table["GV_x"]) and np.allclose(d["pt"], table["GV_pt"])
    assert ft.VERTEX_TARGETS["pt"].fn == "log" and ft.VERTEX_TARGETS["lxy"].fn == "log1p"
    x = np.array([0.0, 1.0, 3.0])
    for fn in (None, "log1p", "sqrt"):
        assert np.allclose(ft.invert_fn(fn, ft.apply_fn(fn, x)), x)
    assert np.allclose(ft.invert_fn("log", ft.apply_fn("log", x[1:])), x[1:])


def test_vertex_classes_and_categories():
    is_b = np.array([1, 0, 0, 0])
    is_d = np.array([0, 1, 1, 0])
    from_hf = np.array([0, 1, 0, 0])
    assert col.vertex_class(is_b, is_d).tolist() == [col.CLASS_B, col.CLASS_C, col.CLASS_C, col.CLASS_OTHER]
    assert col.vertex_category(is_b, is_d, from_hf).tolist() == [col.CATEGORY_B, col.CATEGORY_C_FROM_B, col.CATEGORY_C_PROMPT, col.CATEGORY_OTHER]
    assert col.RESIDUAL_CLASS == col.NUM_CLASSES == len(col.CLASS_NAMES) == 3


def test_track_counts_per_vertex():
    assert col.track_counts_per_vertex(np.array([-1, 0, 2, 2, -1]), 3).tolist() == [1, 0, 2]
    assert col.track_counts_per_vertex(np.array([-1, -1]), 0).tolist() == []
    with pytest.raises(ValueError):
        col.track_counts_per_vertex(np.array([3]), 3)
