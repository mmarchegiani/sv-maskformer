"""The SV plot chain (src/sv/scripts/plot) on a synthetic prediction file with the schema of SVPredictionWriter:
the loader's matching quantities and a full run without a failing section. Torch-free.

    PYTHONPATH=src python -m pytest tests/test_sv_plot_chain.py -q
"""

import json
import pathlib
import sys

import numpy as np
import pytest

ak = pytest.importorskip("awkward")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("matplotlib")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "sv" / "scripts" / "plot"))
sys.path.insert(0, str(REPO / "src"))
import sv_eval_common as common  # noqa: E402
from run_sv_plot_chain import run_chain  # noqa: E402
from sv_eval_common import EvalData  # noqa: E402

common.SAVE_PDF = False
PROBS = ("b", "c", "other", "null")
FIELDS = ("pt", "eta", "sinphi", "cosphi", "lxy", "lz")


def synthetic_predictions(path, n_events=40, seed=3):
    """Events with 0-4 vertices and 30-60 tracks; the first vertex of each event is reconstructed perfectly, the second
    loosely (one track lost, one background track added), the others missed; one fake per odd event."""
    rng = np.random.default_rng(seed)
    gv = {k: [] for k in ("GV_pt", "GV_eta", "GV_phi", "GV_x", "GV_y", "GV_z", "GV_isB", "GV_isD", "GV_fromHF", "GV_class", "GV_category", "GV_nTracks",
                         "GV_lxy", "GV_lz", "GV_l3d", "GV_flight_xy", "GV_nDaughters", "GV_is_target", "GV_n_tracks_selected", "GV_pred_valid", "GV_pred_class",
                         "GV_pred_n_tracks", "GV_pred_n_correct", *[f"GV_pred_prob_{p}" for p in PROBS], *[f"GV_pred_{f}" for f in FIELDS], "GV_pred_phi")}
    tr = {k: [] for k in ("track_pt", "track_eta", "track_hadidx", "track_label", "track_selected", "track_pred_match_idx", "track_pred_slot", "track_pred_prob")}
    fk = {k: [] for k in ("fakeGV_slot", "fakeGV_class", "fakeGV_n_tracks", *[f"fakeGV_prob_{p}" for p in PROBS], *[f"fakeGV_{f}" for f in FIELDS])}
    n_fake = []
    for i in range(n_events):
        k = int(rng.integers(0, 5)); n = int(rng.integers(30, 61))
        is_b = (rng.random(k) < 0.4).astype(np.int32); is_d = 1 - is_b; from_hf = ((rng.random(k) < 0.5) & (is_d == 1)).astype(np.int32)
        cls = np.where(is_b, 0, 1).astype(np.int32); cat = np.where(is_b, 0, np.where(from_hf, 1, 2)).astype(np.int32)
        hadidx = np.full(n, -1, np.int32)
        for j in range(k):
            hadidx[j * 3: j * 3 + 3] = j  # three tracks per vertex
        n_tracks = np.array([3] * k, np.int32)
        pt = rng.uniform(10, 100, k); lxy = rng.uniform(0.01, 3, k)
        match = hadidx.copy(); slot = hadidx.copy(); prob = np.where(hadidx >= 0, 0.9, 0.0).astype(np.float32)
        pred_valid = np.zeros(k, bool); pred_n = np.zeros(k, np.int32); pred_correct = np.zeros(k, np.int32); pred_class = np.full(k, 3, np.int32)
        if k >= 1:
            pred_valid[0] = True; pred_n[0] = 3; pred_correct[0] = 3; pred_class[0] = cls[0]
        if k >= 2:  # loose: 2 of 3 tracks, plus one background track
            pred_valid[1] = True; pred_n[1] = 3; pred_correct[1] = 2; pred_class[1] = cls[1]
            match[5] = -1; slot[5] = -1; prob[5] = 0.0
            match[n - 1] = 1; slot[n - 1] = 1; prob[n - 1] = 0.6
        for j in range(2, k):  # missed: tracks unassigned
            match[j * 3: j * 3 + 3] = -1; slot[j * 3: j * 3 + 3] = -1; prob[j * 3: j * 3 + 3] = 0.0
        probs = np.zeros((k, 4), np.float32)
        for j in range(k):
            probs[j, pred_class[j]] = 0.8; probs[j] += 0.05; probs[j] /= probs[j].sum()
        for key, val in (("GV_pt", pt), ("GV_eta", rng.uniform(-2, 2, k)), ("GV_phi", rng.uniform(-3, 3, k)), ("GV_x", lxy), ("GV_y", np.zeros(k)), ("GV_z", rng.normal(0, 2, k)),
                         ("GV_isB", is_b), ("GV_isD", is_d), ("GV_fromHF", from_hf), ("GV_class", cls), ("GV_category", cat), ("GV_nTracks", n_tracks), ("GV_lxy", lxy),
                         ("GV_lz", rng.normal(0, 1, k)), ("GV_l3d", lxy * 1.2), ("GV_flight_xy", lxy), ("GV_nDaughters", n_tracks), ("GV_is_target", np.ones(k, bool)),
                         ("GV_n_tracks_selected", n_tracks), ("GV_pred_valid", pred_valid), ("GV_pred_class", pred_class), ("GV_pred_n_tracks", pred_n),
                         ("GV_pred_n_correct", pred_correct)):
            gv[key].append(np.asarray(val))
        for p_i, p in enumerate(PROBS):
            gv[f"GV_pred_prob_{p}"].append(probs[:, p_i])
        pred = {"pt": pt * rng.normal(1, 0.1, k), "eta": rng.uniform(-2, 2, k), "sinphi": np.zeros(k), "cosphi": np.ones(k), "lxy": lxy * rng.normal(1, 0.2, k), "lz": rng.normal(0, 1, k)}
        for f in FIELDS:
            gv[f"GV_pred_{f}"].append(np.where(pred_valid, pred[f], np.nan).astype(np.float32))
        gv["GV_pred_phi"].append(np.zeros(k, np.float32))
        for key, val in (("track_pt", rng.uniform(0.8, 20, n)), ("track_eta", rng.uniform(-2.5, 2.5, n)), ("track_hadidx", hadidx),
                         ("track_label", np.where(hadidx >= 0, 2, 0)), ("track_selected", np.ones(n, bool)), ("track_pred_match_idx", match),
                         ("track_pred_slot", slot), ("track_pred_prob", prob)):
            tr[key].append(np.asarray(val))
        nf = 1 if i % 2 else 0
        fk["fakeGV_slot"].append(np.full(nf, 7, np.int64)); fk["fakeGV_class"].append(np.full(nf, 1, np.int64)); fk["fakeGV_n_tracks"].append(np.full(nf, 2, np.int64))
        for p_i, p in enumerate(PROBS):
            fk[f"fakeGV_prob_{p}"].append(np.full(nf, 0.25, np.float32))
        for f in FIELDS:
            fk[f"fakeGV_{f}"].append(np.full(nf, 1.0, np.float32))
        n_fake.append(nf)
    events = ak.zip({
        "GV": ak.zip({k: ak.Array(v) for k, v in gv.items()}, depth_limit=1),
        "track": ak.zip({k: ak.Array(v) for k, v in tr.items()}, depth_limit=1),
        "fakeGV": ak.zip({k: ak.Array(v) for k, v in fk.items()}, depth_limit=1),
        "n_fake_vertices": ak.Array(np.asarray(n_fake, np.int64)),
        "event_number": ak.Array(np.arange(n_events, dtype=np.int64)),
    }, depth_limit=1)
    pq.write_table(ak.to_arrow_table(events, extensionarray=False), path)
    return path


def test_loader_matching_quantities(tmp_path):
    d = EvalData(synthetic_predictions(tmp_path / "pred.parquet"))
    assert d.n_events == 40 and d.regressed_fields == sorted(["pt", "eta", "sinphi", "cosphi", "phi", "lxy", "lz"])
    first = np.r_[0, np.cumsum(d.e_n_vertices)[:-1]][d.e_n_vertices >= 1]
    assert d.v_perfect[first].all() and d.v_loose[first].all()
    second = first[d.e_n_vertices[d.e_n_vertices >= 1] >= 2] + 1
    assert d.v_loose[second].all() and not d.v_perfect[second].any()
    assert np.allclose(d.v_recall[second], 2 / 3) and np.allclose(d.v_purity[second], 2 / 3)
    assert int(d.e_n_fakes.sum()) == 20 and len(d.f_class) == 20
    assert d.t_match.min() == -1 and (d.t_selected.all())


def test_chain_runs_without_failures(tmp_path):
    path = synthetic_predictions(tmp_path / "pred.parquet")
    outdir = tmp_path / "plots"
    result = run_chain(str(path), str(outdir), max_events=30)
    assert result["failures"] == {}, result["failures"]
    metrics = json.loads((outdir / "metrics.json").read_text())
    assert metrics["n_events"] == 30
    assert 0 < metrics["efficiency"]["loose"]["efficiency"] <= 1 and metrics["efficiency"]["perfect"]["efficiency"] < metrics["efficiency"]["loose"]["efficiency"]
    assert metrics["fakes"]["n_fake_vertices"] == 15
    assert "pt" in metrics["regression"] and "confusion_rows_true_cols_pred" in metrics["classification"]
    assert metrics["tracks"]["assigned_to_right_vertex"] > 0.5
    for section in ("efficiency", "fakes", "classification", "regression", "tracks"):
        assert any((outdir / section).glob("*.png")), section
    assert (outdir / "summary.txt").read_text().startswith("SV MaskFormer evaluation")
