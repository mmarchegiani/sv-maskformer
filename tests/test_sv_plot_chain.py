"""The SV plot chain (src/sv/scripts/plot) on a synthetic prediction file with the schema of SVPredictionWriter:
the loader's matching quantities, a full run without a failing section, the resolution section (the absolute and
relative residuals of every regressed field, its selection fallback, its standalone script) and the working-point
scan (fake rate vs efficiency, roc/). Torch-free.

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
import plot_sv_eval as ev  # noqa: E402
import plot_sv_resolution as resolution  # noqa: E402
from run_sv_plot_chain import run_chain  # noqa: E402
from sv_eval_common import EvalData  # noqa: E402

common.SAVE_PDF = False
PROBS = ("B", "BtoC", "C", "Other", "null")
FIELDS = ("pt", "eta", "sinphi", "cosphi", "lxy", "lz")


N_SLOTS = 8


def synthetic_predictions(path, n_events=40, seed=3, slots=True):
    """Events with 0-4 vertices and 30-60 tracks; the first vertex of each event is reconstructed perfectly, the second
    loosely (one track lost, one background track added), the others missed; one fake per odd event.

    With `slots`, the raw output of the 8 query slots behind those predictions (the `slot` collection): slot j holds
    vertex j; the missed vertices have the right tracks (mask probability 0.7) but p(null) = 0.8; the lost track of the
    loose vertex has a mask probability of 0.3, its background track 0.6; the fake is slot 7, with two background
    tracks (0.8) and p(null) = 0.25; the unused slots have p(null) = 0.97 and no track."""
    rng = np.random.default_rng(seed)
    gv = {k: [] for k in ("GV_pt", "GV_eta", "GV_phi", "GV_x", "GV_y", "GV_z", "GV_isB", "GV_isD", "GV_fromHF", "GV_class", "GV_nTracks",
                         "GV_lxy", "GV_lz", "GV_l3d", "GV_flight_xy", "GV_nDaughters", "GV_is_target", "GV_n_tracks_selected", "GV_pred_valid", "GV_pred_class",
                         "GV_pred_n_tracks", "GV_pred_n_correct", *[f"GV_pred_prob_{p}" for p in PROBS], *[f"GV_pred_{f}" for f in FIELDS], "GV_pred_phi")}
    tr = {k: [] for k in ("track_pt", "track_eta", "track_hadidx", "track_label", "track_selected", "track_pred_match_idx", "track_pred_slot", "track_pred_prob")}
    sl = {k: [] for k in ("slot_prob_null", "slot_target_idx", "slot_active", "slot_pred_valid", "slot_n_tracks", "slot_n_correct", "slot_mask_prob", "slot_truth_mask")}
    fk = {k: [] for k in ("fakeGV_slot", "fakeGV_class", "fakeGV_n_tracks", *[f"fakeGV_prob_{p}" for p in PROBS], *[f"fakeGV_{f}" for f in FIELDS])}
    n_fake = []
    for i in range(n_events):
        k = int(rng.integers(0, 5)); n = int(rng.integers(30, 61))
        is_b = (rng.random(k) < 0.4).astype(np.int32); is_d = 1 - is_b; from_hf = ((rng.random(k) < 0.5) & (is_d == 1)).astype(np.int32)
        cls = np.where(is_b, 0, np.where(from_hf, 1, 2)).astype(np.int32)
        hadidx = np.full(n, -1, np.int32)
        for j in range(k):
            hadidx[j * 3: j * 3 + 3] = j  # three tracks per vertex
        n_tracks = np.array([3] * k, np.int32)
        pt = rng.uniform(10, 100, k); lxy = rng.uniform(0.01, 3, k)
        match = hadidx.copy(); slot = hadidx.copy(); prob = np.where(hadidx >= 0, 0.9, 0.0).astype(np.float32)
        pred_valid = np.zeros(k, bool); pred_n = np.zeros(k, np.int32); pred_correct = np.zeros(k, np.int32); pred_class = np.full(k, 4, np.int32)
        if k >= 1:
            pred_valid[0] = True; pred_n[0] = 3; pred_correct[0] = 3; pred_class[0] = cls[0]
        if k >= 2:  # loose: 2 of 3 tracks, plus one background track
            pred_valid[1] = True; pred_n[1] = 3; pred_correct[1] = 2; pred_class[1] = cls[1]
            match[5] = -1; slot[5] = -1; prob[5] = 0.0
            match[n - 1] = 1; slot[n - 1] = 1; prob[n - 1] = 0.6
        for j in range(2, k):  # missed: tracks unassigned
            match[j * 3: j * 3 + 3] = -1; slot[j * 3: j * 3 + 3] = -1; prob[j * 3: j * 3 + 3] = 0.0
        probs = np.zeros((k, 5), np.float32)
        for j in range(k):
            probs[j, pred_class[j]] = 0.8; probs[j] += 0.05; probs[j] /= probs[j].sum()
        for key, val in (("GV_pt", pt), ("GV_eta", rng.uniform(-2, 2, k)), ("GV_phi", rng.uniform(-3, 3, k)), ("GV_x", lxy), ("GV_y", np.zeros(k)), ("GV_z", rng.normal(0, 2, k)),
                         ("GV_isB", is_b), ("GV_isD", is_d), ("GV_fromHF", from_hf), ("GV_class", cls), ("GV_nTracks", n_tracks), ("GV_lxy", lxy),
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
        p_null = np.full(N_SLOTS, 0.97, np.float32); mask_prob = np.full((N_SLOTS, n), 0.01, np.float32); truth = np.zeros((N_SLOTS, n), bool)
        for j in range(k):
            truth[j] = hadidx == j
            mask_prob[j, truth[j]] = 0.9 if j < 2 else 0.7
            p_null[j] = 0.05 if j < 2 else 0.8
        if k >= 2:
            mask_prob[1, 5] = 0.3; mask_prob[1, n - 1] = 0.6
        if nf:
            p_null[7] = 0.25; mask_prob[7, n - 3: n - 1] = 0.8
        on = mask_prob >= 0.5
        for key, val in (("slot_prob_null", p_null), ("slot_target_idx", np.where(np.arange(N_SLOTS) < k, np.arange(N_SLOTS), -1)),
                         ("slot_active", np.ones(N_SLOTS, bool)), ("slot_pred_valid", p_null < 0.5), ("slot_n_tracks", on.sum(1)),
                         ("slot_n_correct", (on & truth).sum(1)), ("slot_mask_prob", mask_prob), ("slot_truth_mask", truth)):
            sl[key].append(np.asarray(val))
    events = ak.zip({
        "GV": ak.zip({k: ak.Array(v) for k, v in gv.items()}, depth_limit=1),
        "track": ak.zip({k: ak.Array(v) for k, v in tr.items()}, depth_limit=1),
        "fakeGV": ak.zip({k: ak.Array(v) for k, v in fk.items()}, depth_limit=1),
        **({"slot": ak.zip({k: ak.Array([a.tolist() for a in v]) for k, v in sl.items()}, depth_limit=1)} if slots else {}),
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
    for section in ("efficiency", "fakes", "classification", "regression", "tracks", "roc", "resolution"):
        assert any((outdir / section).glob("*.png")), section
    summary = (outdir / "summary.txt").read_text()
    assert summary.startswith("SV MaskFormer evaluation") and "Resolution (loose-matched target vertices" in summary
    res = metrics["resolution"]
    assert res["selection"] == res["selection_requested"] == "loose" and res["n_vertices"] == metrics["regression"]["n_vertices"] > 0
    assert res["fields"] == ["pt", "eta", "phi", "sinphi", "cosphi", "lxy", "lz"]
    for field in res["fields"]:
        for kind in ("abs", "rel"):
            assert (outdir / "resolution" / f"residual_{kind}_{field}.png").is_file()
            stats = res[field][kind]
            assert stats["n"] == res["n_vertices"] and set(stats) >= {"median", "mean", "sigma68", "std", "window", "beyond_window"}
            assert stats["window"][0] == -stats["window"][1] < 0 and 0 <= stats["beyond_window"] <= 1
    assert (outdir / "resolution" / "overview.png").is_file()
    # the synthetic predictions: pT and lxy smeared around the truth, eta drawn independently
    assert abs(res["pt"]["rel"]["median"]) < 0.1 and res["pt"]["rel"]["sigma68"] < 0.3
    assert res["eta"]["abs"]["sigma68"] > res["pt"]["rel"]["sigma68"]


def test_resolution_residuals():
    true, pred = np.array([10.0, 0.0, 3.0, -3.0]), np.array([12.0, 1.0, -3.0, 3.0])
    assert np.allclose(resolution.residual(true, pred, "pt", "abs"), [2.0, 1.0, -6.0, 6.0])
    rel = resolution.residual(true, pred, "pt", "rel")
    assert np.isnan(rel[1]) and np.allclose(rel[[0, 2, 3]], [0.2, -2.0, -2.0])
    wrapped = resolution.residual(true, pred, "phi", "abs")  # -3 - 3 wraps to 2 pi - 6, 3 - (-3) to 6 - 2 pi
    assert np.allclose(wrapped, [2.0, 1.0, 2 * np.pi - 6, 6 - 2 * np.pi])
    assert np.allclose(resolution.residual_window(np.array([0.1, 0.1, 0.1]), 5.0), (-0.15, 0.15))  # zero width: 1.5 |median|
    lo, hi = resolution.residual_window(np.random.default_rng(0).normal(0.0, 1.0, 10000), 5.0)
    assert lo == -hi and 4.5 < hi < 5.5
    with pytest.raises(ValueError):
        resolution.residual(true, pred, "pt", "squared")


def test_resolution_selection_fallback(tmp_path):
    d = EvalData(synthetic_predictions(tmp_path / "pred.parquet"))
    mask, used = resolution.select_vertices(d, "perfect")
    assert used == "perfect" and mask.sum() == (d.v_perfect & (d.v_n_sel >= 2)).sum() > 0
    d.v_loose[:] = False
    d.v_perfect[:] = False
    metrics = {}
    m = resolution.resolution_section(d, str(tmp_path / "plots"), metrics, selection="loose")
    assert metrics["resolution"] is m and m["selection"] == "found" and m["selection_requested"] == "loose"
    assert m["n_vertices"] == int((d.v_is_target & (d.v_n_sel >= 2) & d.v_pred_valid).sum()) > 0
    assert (tmp_path / "plots" / "resolution" / "residual_rel_lxy.png").is_file()
    assert "no loose-matched vertex" in "\n".join(resolution.summary_lines(m))
    with pytest.raises(ValueError):
        resolution.select_vertices(d, "all")


def test_resolution_standalone_script(tmp_path, capsys):
    path = synthetic_predictions(tmp_path / "pred.parquet")
    outdir = tmp_path / "plots"
    assert resolution.main(["--input", str(path), "--outdir", str(outdir), "--selection", "targets", "--max-events", "20", "--range-sigmas", "3"]) == 0
    m = json.loads((outdir / "resolution" / "metrics.json").read_text())
    assert m["selection"] == "targets" and m["range_sigmas"] == 3.0
    assert 0 < m["pt"]["abs"]["n"] < m["n_vertices"]  # the missed vertices of the synthetic file carry no prediction (NaN)
    assert len(list((outdir / "resolution").glob("residual_*.png"))) == 2 * len(m["fields"])
    assert "Resolution (all target vertices" in capsys.readouterr().out


def _scan_at(scan, t, m, k):
    i = np.flatnonzero(np.isclose(scan["score_cut"], t) & np.isclose(scan["mask_threshold"], m) & (scan["min_tracks"] == k))
    assert len(i) == 1
    return {name: float(values[i[0]]) for name, values in scan.items()}


def test_working_point_scan(tmp_path):
    d = EvalData(synthetic_predictions(tmp_path / "pred.parquet"))
    assert d.has_slots and len(d.s_score) == 40 * N_SLOTS and len(d.p_prob) == len(d.p_slot) == len(d.p_truth)
    sel = d.v_is_target & (d.v_n_sel >= 2)
    n_targets, n_loose = int(sel.sum()), int(d.v_loose[sel].sum())

    # the model's working point reproduces the loose efficiency of the efficiency section; its fakes are the 20 fake slots
    wp = ev.model_working_point(d)
    assert np.isclose(wp["efficiency"], n_loose / n_targets)
    assert wp["n_fake"] == 20 and np.isclose(wp["fake_rate"], 20 / (20 + n_loose)) and np.isclose(wp["fakes_per_event"], 0.5)
    assert ev.model_working_point(d, min_tracks=3)["n_fake"] == 0  # the fakes have two tracks

    scan = ev.scan_working_points(d, score_cuts=[0.0, 0.5, 0.9, 0.96], mask_thresholds=[0.2, 0.5, 0.65], min_tracks=(1, 2, 3))
    assert len(scan["efficiency"]) == 4 * 3 * 3
    # the same cuts as the model: the scan agrees with the working point
    same = _scan_at(scan, 0.5, 0.5, 1)
    assert all(np.isclose(same[name], wp[name]) for name in wp)
    # no score cut: the missed vertices (right tracks, high p(null)) are found too, the fakes stay
    loose_all = _scan_at(scan, 0.0, 0.5, 1)
    assert loose_all["efficiency"] == 1.0 and np.isclose(loose_all["fake_rate"], 20 / (20 + n_targets))
    # a tight score cut removes the fakes (score 0.75) and keeps the found vertices (0.95); a tighter one removes all
    assert _scan_at(scan, 0.9, 0.5, 1)["n_fake"] == 0 and np.isclose(_scan_at(scan, 0.9, 0.5, 1)["efficiency"], wp["efficiency"])
    empty = _scan_at(scan, 0.96, 0.5, 1)
    assert empty["efficiency"] == 0 and np.isnan(empty["fake_rate"])
    # the mask threshold changes the quality, not the number of vertices: at 0.65 the loose vertices lose their background
    # track (recall and purity 2/3 and 1), at 0.2 they recover the lost one and keep the background one (1 and 3/4)
    for m in (0.2, 0.65):
        assert np.isclose(_scan_at(scan, 0.5, m, 1)["efficiency"], wp["efficiency"])
    # the efficiency and the number of predicted vertices can only decrease with the score cut and with min_tracks
    for m in (0.2, 0.5, 0.65):
        for k in (1, 2, 3):
            rows = [_scan_at(scan, t, m, k) for t in (0.0, 0.5, 0.9, 0.96)]
            assert all(a["efficiency"] >= b["efficiency"] and a["n_matched"] + a["n_fake"] >= b["n_matched"] + b["n_fake"] for a, b in zip(rows, rows[1:]))
        assert _scan_at(scan, 0.0, m, 1)["n_fake"] >= _scan_at(scan, 0.0, m, 2)["n_fake"] >= _scan_at(scan, 0.0, m, 3)["n_fake"]


def test_pareto_front():
    eff = np.array([0.9, 0.8, 0.8, 0.5, 0.95, 0.3, 0.7])
    fr = np.array([0.2, 0.1, 0.3, 0.05, 0.5, 0.05, np.nan])
    front = ev.pareto_front(eff, fr)
    assert front.tolist() == [3, 1, 0, 4]  # by increasing efficiency; 2 and 5 are dominated, 6 has no predicted vertex
    for i in front:  # none of them is dominated
        assert not any((eff[j] >= eff[i]) and (fr[j] <= fr[i]) and ((eff[j] > eff[i]) or (fr[j] < fr[i])) for j in range(len(eff)) if np.isfinite(fr[j]))


def test_roc_section_in_the_chain(tmp_path):
    result = run_chain(str(synthetic_predictions(tmp_path / "pred.parquet")), str(tmp_path / "plots"))
    assert result["failures"] == {}, result["failures"]
    m = result["metrics"]
    roc = m["roc"]
    assert np.isclose(roc["working_point"]["min_tracks_1"]["efficiency"], m["efficiency"]["loose"]["efficiency"])
    assert roc["n_configurations"] == len(ev.ROC_SCORE_CUTS) * len(ev.ROC_MASK_THRESHOLDS) * len(ev.ROC_MIN_TRACKS)
    front = roc["pareto_front"]
    assert len(front) >= 1 and all(a["efficiency"] < b["efficiency"] and a["fake_rate"] < b["fake_rate"] for a, b in zip(front, front[1:]))
    for name in ("fake_rate_vs_efficiency", "fake_rate_vs_efficiency_score_scan", "fake_rate_vs_efficiency_mask_thresholds"):
        assert (tmp_path / "plots" / "roc" / f"{name}.png").is_file()
    header = (tmp_path / "plots" / "roc" / "scan.csv").read_text().splitlines()
    assert header[0].startswith("score_cut,mask_threshold,min_tracks") and len(header) == roc["n_configurations"] + 1
    assert "Working point (fake rate" in (tmp_path / "plots" / "summary.txt").read_text()


def test_dedup_pass():
    """One event, 5 selected tracks, a target on tracks 0-2: slot 0 is its Hungarian match (tracks 0, 1, score 0.8),
    slot 1 an unmatched duplicate with all three tracks and a higher score, slot 2 a fake on tracks 3, 4."""
    from types import SimpleNamespace

    on = np.array([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=bool)
    truth = np.zeros_like(on)
    truth[0, :3] = True
    d = SimpleNamespace(n_events=1, s_event=np.zeros(3, dtype=np.int64), s_score=np.array([0.8, 0.9, 0.7]),
                        s_vertex=np.array([0, -1, -1]), v_n_sel=np.array([3]), p_slot=np.repeat(np.arange(3), 5),
                        p_prob=np.where(on, 0.9, 0.1).reshape(-1), p_truth=truth.reshape(-1))
    cand = np.ones(3, dtype=bool)
    # no removal: the duplicate (higher score) takes the target, the Hungarian match becomes a fake
    s, oc = ev.dedup_pass(d, 1.0, cand)
    assert s.tolist() == [0.9, 0.8, 0.7] and oc.tolist() == [1, 0, 0]
    # removal: the lower-score copy shares 2 of its 2 tracks with the kept duplicate and is dropped
    s, oc = ev.dedup_pass(d, 0.5, cand)
    assert s.tolist() == [0.9, 0.7] and oc.tolist() == [1, 0]
    # a target below EVAL_MIN_TRACKS: matched vertices are ignored (-1), not fakes
    d.v_n_sel = np.array([1])
    assert ev.dedup_pass(d, 0.5, cand)[1].tolist() == [-1, 0]
    d.v_n_sel = np.array([3])
    # min_tracks: the 2-track vertices do not qualify at 3
    assert ev.dedup_pass(d, 1.0, cand, min_tracks=3)[1].tolist() == [1]


def test_dedup_section_in_the_chain(tmp_path):
    result = run_chain(str(synthetic_predictions(tmp_path / "pred.parquet")), str(tmp_path / "plots"))
    assert result["failures"] == {}, result["failures"]
    dd = result["metrics"]["dedup"]
    assert set(dd["working_point"]) == {"none" if o >= 1 else f"{o:g}" for o in ev.DEDUP_OVERLAPS}
    for name in ("fake_rate_vs_efficiency", "fakes_per_event_vs_efficiency"):
        assert (tmp_path / "plots" / "dedup" / f"{name}.png").is_file()
    assert "Duplicate removal" in (tmp_path / "plots" / "summary.txt").read_text()


def test_roc_section_skipped_without_slots(tmp_path):
    result = run_chain(str(synthetic_predictions(tmp_path / "pred.parquet", slots=False)), str(tmp_path / "plots"))
    assert result["failures"] == {}, result["failures"]
    assert "skipped" in result["metrics"]["roc"] and not (tmp_path / "plots" / "roc").exists()
    assert "Working point" not in (tmp_path / "plots" / "summary.txt").read_text()
