"""sv.scripts.dataset.convert_root and compute_norm_par on a synthetic BTV ntuple (written with uproot; torch-free):
the parquet layout, the generated primary vertex, the candidate vertex selection, the derived truth fields, the split
and the fitted scale dicts.

    PYTHONPATH=src python -m pytest tests/test_sv_convert.py -q
"""

import pathlib
import sys

import numpy as np
import pytest

ak = pytest.importorskip("awkward")
uproot = pytest.importorskip("uproot")
pq = pytest.importorskip("pyarrow.parquet")
yaml = pytest.importorskip("yaml")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sv.dataset.collections import (  # noqa: E402
    OPTIONAL_COLLECTIONS,
    PV_FIELDS,
    TRACK_BRANCHES,
    VERTEX_BRANCHES,
    VERTEX_DERIVED,
    select_daughters,
    select_vertices,
    vertex_class,
)
from sv.scripts.dataset import compute_norm_par as norm  # noqa: E402
from sv.scripts.dataset import convert_root as conv  # noqa: E402

N_EVENTS = 12
#: failure modes of the synthetic vertices that must fail the candidate selection
FAIL_PT, FAIL_ETA, FAIL_DAUGHTERS = 0, 1, 2


def synthetic_events(seed=0):
    """A small Events tree: per event 0-4 vertices (the first prompt), 20-40 tracks, 30% of them pointing to a vertex,
    the `GVDaughters` of every vertex and the `trackPair` collection of the first 8 tracks. Some vertices fail the
    candidate selection (pT < 10, |eta| > 2.5 or a single good daughter): `n_gv` counts the vertices that pass."""
    rng = np.random.default_rng(seed)
    events = {}
    n_gv_raw = rng.integers(0, 5, N_EVENTS)
    n_gv = np.zeros(N_EVENTS, dtype=np.int64)
    n_tr = rng.integers(20, 41, N_EVENTS)
    scalars = {"run": np.ones(N_EVENTS, np.uint32), "luminosityBlock": np.arange(N_EVENTS, dtype=np.uint32), "event": np.arange(N_EVENTS, dtype=np.uint64)}
    pv = rng.normal([0.03, -0.02, 0.0], [0.001, 0.001, 3.0], (N_EVENTS, 3))
    gv = {name: [] for name in VERTEX_BRANCHES}
    dau = {name: [] for name in OPTIONAL_COLLECTIONS["GVDaughters"]}
    pair = {name: [] for name in OPTIONAL_COLLECTIONS["trackPair"]}
    tr = {name: [] for name in TRACK_BRANCHES}
    gen = {"GenPart_status": [], "GenPart_vx": [], "GenPart_vy": [], "GenPart_vz": []}
    forced = 0  # the first events with >= 2 vertices get vertex 1 failing in each mode in turn
    for i in range(N_EVENTS):
        k, n = int(n_gv_raw[i]), int(n_tr[i])
        is_b = (rng.random(k) < 0.4).astype(np.int32)
        is_d = 1 - is_b
        from_hf = ((rng.random(k) < 0.5) & (is_d == 1)).astype(np.int32)
        pos = pv[i] + rng.normal(0, [0.3, 0.3, 0.5], (k, 3))
        fail = np.where(rng.random(k) < 0.3, rng.integers(0, 3, k), -1)
        if k:
            fail[0] = -1  # the first vertex always passes
        if k >= 2 and forced < 3:
            fail[1] = (FAIL_PT, FAIL_DAUGHTERS, FAIL_ETA)[forced]
            forced += 1
        n_gv[i] = int((fail < 0).sum())
        pt = np.where(fail == FAIL_PT, rng.uniform(5, 9.9, k), rng.uniform(10.1, 100, k)).astype(np.float32)
        eta = np.where(fail == FAIL_ETA, rng.choice([-1, 1], k) * rng.uniform(2.6, 3.0, k), rng.uniform(-2.4, 2.4, k)).astype(np.float32)
        n_dau = rng.integers(2, 5, k).astype(np.int32)
        gv["GV_Hadron_pdgId"].append(np.where(is_b, 511, 421).astype(np.int32))
        gv["GV_isB"].append(is_b); gv["GV_isD"].append(is_d); gv["GV_fromHF"].append(from_hf)
        gv["GV_toHF"].append(np.zeros(k, np.int32)); gv["GV_pdgClass"].append(np.where(is_b, 511, 421).astype(np.int32))
        gv["GV_nDaughters"].append(n_dau)
        gv["GV_pt"].append(pt); gv["GV_eta"].append(eta)
        gv["GV_phi"].append(rng.uniform(-np.pi, np.pi, k).astype(np.float32))
        for j, a in enumerate("xyz"):
            gv[f"GV_{a}"].append(pos[:, j].astype(np.float32))
            gv[f"GV_{a}_i"].append(np.full(k, pv[i][j], np.float32))
        gv["GV_maxDaughterPairDeltaR"].append(rng.uniform(0, 0.5, k).astype(np.float32))
        # the daughters: good ones, except for a FAIL_DAUGHTERS vertex whose daughters but the first are neutral,
        # soft, forward or unstable in turn
        d_idx = np.repeat(np.arange(k), n_dau).astype(np.int32)
        nd = len(d_idx)
        d_charge = rng.choice([-1, 1], nd).astype(np.int32); d_pt = rng.uniform(1, 20, nd).astype(np.float32)
        d_eta = rng.uniform(-2.4, 2.4, nd).astype(np.float32); d_pdg = rng.choice([211, 321, 2212, 11, 13], nd).astype(np.int32) * d_charge
        rank = np.concatenate([np.arange(m) for m in n_dau]) if k else np.zeros(0, np.int64)
        bad = (fail[d_idx] == FAIL_DAUGHTERS) & (rank > 0) if k else np.zeros(0, bool)
        for b in np.flatnonzero(bad):
            mode = int(rank[b]) % 4
            if mode == 0: d_charge[b] = 0
            elif mode == 1: d_pt[b] = 0.5
            elif mode == 2: d_eta[b] = 2.8
            else: d_pdg[b] = 411
        dau["GVDaughters_hadronIndex"].append(d_idx); dau["GVDaughters_charge"].append(d_charge); dau["GVDaughters_pt"].append(d_pt)
        dau["GVDaughters_eta"].append(d_eta); dau["GVDaughters_pdgId"].append(d_pdg); dau["GVDaughters_phi"].append(rng.uniform(-np.pi, np.pi, nd).astype(np.float32))
        dau["GVDaughters_statusFlags"].append(np.full(nd, 12352, np.int32))
        for j, a in enumerate("xyz"):
            dau[f"GVDaughters_v{a}"].append(pos[d_idx, j].astype(np.float32) if k else np.zeros(0, np.float32))
        hadidx = np.where(rng.random(n) < 0.3, rng.integers(0, max(k, 1), n), -1).astype(np.int32) if k else np.full(n, -1, np.int32)
        # track pairs among the first 8 tracks, with the raw edge label
        i1, i2 = np.triu_indices(8, k=1)
        pair["trackPair_trk1Idx"].append(i1.astype(np.int32)); pair["trackPair_trk2Idx"].append(i2.astype(np.int32))
        pair["trackPair_edgeLabel"].append(((hadidx[i1] == hadidx[i2]) & (hadidx[i1] >= 0)).astype(np.int32))
        for name in OPTIONAL_COLLECTIONS["trackPair"]:
            if name not in ("trackPair_trk1Idx", "trackPair_trk2Idx", "trackPair_edgeLabel"):
                pair[name].append(rng.uniform(0, 1, len(i1)).astype(np.float32))
        pt = rng.uniform(0.8, 30, n).astype(np.float32)
        tr["track_pt"].append(pt); tr["track_eta"].append(rng.uniform(-2.5, 2.5, n).astype(np.float32)); tr["track_phi"].append(rng.uniform(-np.pi, np.pi, n).astype(np.float32))
        tr["track_energy"].append(pt * 2); tr["track_px"].append(pt); tr["track_py"].append(pt); tr["track_pz"].append(pt)
        tr["track_charge"].append(rng.choice([-1, 1], n).astype(np.int32)); tr["track_qoverp"].append(rng.uniform(-1, 1, n).astype(np.float32)); tr["track_qdotp"].append(pt)
        tr["track_nHits"].append(rng.integers(3, 30, n).astype(np.int32)); tr["track_numberOfPixelHits"].append(rng.integers(0, 8, n).astype(np.int32))
        tr["track_numberOfStripHits"].append(rng.integers(0, 25, n).astype(np.int32)); tr["track_chi2"].append(rng.integers(0, 50, n).astype(np.float32))
        tr["track_isHighPurity"].append(rng.random(n) < 0.99); tr["track_passSeed"].append(rng.integers(0, 2, n).astype(np.int32))
        tr["track_dz"].append(rng.normal(0, 3, n).astype(np.float32)); tr["track_dzSignificance"].append(rng.normal(0, 100, n).astype(np.float32))
        ip = np.abs(rng.normal(0, 0.1, n)).astype(np.float32)
        tr["track_ip2dValue"].append(ip); tr["track_ip2dSignificance"].append(ip * 100); tr["track_ip3dValue"].append(ip * 2)
        sig = ip * 100
        sig[:2] = -9999.0  # the "not available" marker
        tr["track_ip3dSignificance"].append(sig); tr["track_timeSig"].append(np.full(n, -9999.0, np.float32))
        tr["track_hadidx"].append(hadidx); tr["track_label"].append(np.where(hadidx >= 0, 2, rng.integers(0, 2, n)).astype(np.int32))
        gen["GenPart_status"].append(np.array([4, 4, 21, 21, 22], np.int32))
        for j, a in enumerate("xyz"):
            gen[f"GenPart_v{a}"].append(np.array([0, 0, pv[i][j], pv[i][j], pv[i][j]], np.float32))
    events.update(scalars)
    for d in (gv, tr, gen, dau, pair):
        for name, arrays in d.items():
            events[name] = ak.Array(arrays)
    return events, pv, n_gv, n_tr


def raw_vertex_selection(events):
    """The candidate selection recomputed on the raw synthetic branches: the flat pass mask and the raw counts."""
    n_raw = ak.to_numpy(ak.num(events["GV_pt"], axis=1)).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(n_raw)])
    d_event = np.repeat(np.arange(len(n_raw)), ak.to_numpy(ak.num(events["GVDaughters_pt"], axis=1)))
    d_pass = select_daughters(*(ak.to_numpy(ak.flatten(events[f"GVDaughters_{b}"])) for b in ("charge", "pt", "eta", "pdgId")))
    d_idx = ak.to_numpy(ak.flatten(events["GVDaughters_hadronIndex"]))
    n_sel = np.bincount(offsets[d_event[d_pass]] + d_idx[d_pass], minlength=int(n_raw.sum()))
    passing = select_vertices(ak.to_numpy(ak.flatten(events["GV_pt"])), ak.to_numpy(ak.flatten(events["GV_eta"])), n_sel)
    return passing, n_raw, n_sel


@pytest.fixture(scope="module")
def root_file(tmp_path_factory):
    events, pv, n_gv, n_tr = synthetic_events()
    path = tmp_path_factory.mktemp("sv") / "ttbar_synth.root"
    with uproot.recreate(path) as f:
        f["Events"] = events
    return path, pv, n_gv, n_tr


def test_convert_layout_pv_and_derived_fields(root_file, tmp_path):
    path, pv, n_gv, n_tr = root_file
    assert conv.main([str(path), "--outdir", str(tmp_path), "--step", "5"]) == 0
    out = tmp_path / "ttbar_synth.parquet"
    df = ak.from_arrow(pq.read_table(out))
    assert set(df.fields) == {"run", "luminosityBlock", "event", *PV_FIELDS, "track", "GV"}
    assert set(df["track"].fields) == set(TRACK_BRANCHES) and set(df["GV"].fields) == set(VERTEX_BRANCHES) | set(VERTEX_DERIVED)
    assert np.allclose(ak.to_numpy(df["PV_x"]), pv[:, 0], atol=1e-5) and np.allclose(ak.to_numpy(df["PV_z"]), pv[:, 2], atol=1e-4)
    assert ak.to_numpy(ak.num(df["GV"]["GV_pt"], axis=1)).tolist() == n_gv.tolist()
    assert ak.to_numpy(ak.num(df["track"]["track_pt"], axis=1)).tolist() == n_tr.tolist()
    gv = df["GV"]
    x, y = ak.to_numpy(ak.flatten(gv["GV_x"])), ak.to_numpy(ak.flatten(gv["GV_y"]))
    pvx = np.repeat(pv[:, 0], n_gv); pvy = np.repeat(pv[:, 1], n_gv)
    assert np.allclose(ak.to_numpy(ak.flatten(gv["GV_lxy"])), np.hypot(x - pvx, y - pvy), atol=1e-4)
    assert np.allclose(ak.to_numpy(ak.flatten(gv["GV_flight_xy"])), ak.to_numpy(ak.flatten(gv["GV_lxy"])), atol=1e-4)  # prompt: same origin
    # GV_nTracks counts the tracks pointing to each vertex
    for i in range(N_EVENTS):
        hadidx = ak.to_numpy(df["track"]["track_hadidx"][i])
        expected = np.bincount(hadidx[hadidx >= 0], minlength=n_gv[i])[: n_gv[i]] if n_gv[i] else []
        assert ak.to_numpy(gv["GV_nTracks"][i]).tolist() == list(expected)
    cls, is_b, is_d, from_hf = (ak.to_numpy(ak.flatten(gv[b])) for b in ("GV_class", "GV_isB", "GV_isD", "GV_fromHF"))
    assert np.array_equal(cls, vertex_class(is_b, is_d, from_hf)) and set(cls.tolist()) == {0, 1, 2}  # B, BtoC, C in the synthetic file


def test_convert_split_and_optional_collections(root_file, tmp_path):
    path = root_file[0]
    assert conv.main([str(path), "--outdir", str(tmp_path), "--split", "0.5", "0.25", "0.25", "--max-events", "8"]) == 0
    counts = {name: pq.read_metadata(tmp_path / name / "ttbar_synth.parquet").num_rows for name in ("train", "val", "test")}
    assert counts == {"train": 4, "val": 2, "test": 2}
    with pytest.raises(ValueError):
        conv.split_counts(10, [0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="lacks the branches"):
        conv.main([str(path), "--outdir", str(tmp_path / "x"), "--keep", "RejectedGV"])


def test_candidate_vertex_selection(root_file, tmp_path):
    """Vertices failing pT > 10, |eta| < 2.5 or >= 2 good daughters are dropped; track_hadidx, the daughters and the
    pair labels are renumbered accordingly."""
    path, pv, n_gv, n_tr = root_file
    events = synthetic_events()[0]
    passing, n_raw, n_sel = raw_vertex_selection(events)
    assert n_raw.sum() > n_gv.sum() and (~passing).sum() >= 3  # every failure mode is exercised
    assert conv.main([str(path), "--outdir", str(tmp_path), "--keep", "GVDaughters", "--keep", "trackPair"]) == 0
    df = ak.from_arrow(pq.read_table(tmp_path / "ttbar_synth.parquet"))
    gv, tr = df["GV"], df["track"]
    assert ak.to_numpy(ak.num(gv["GV_pt"], axis=1)).tolist() == n_gv.tolist()
    assert "GV_nSelectedDaughters" in gv.fields
    kept_pt, kept_eta, kept_sel = (ak.to_numpy(ak.flatten(gv[b])) for b in ("GV_pt", "GV_eta", "GV_nSelectedDaughters"))
    assert (kept_pt > 10).all() and (np.abs(kept_eta) < 2.5).all() and (kept_sel >= 2).all()
    assert np.allclose(kept_pt, ak.to_numpy(ak.flatten(events["GV_pt"]))[passing]) and np.array_equal(kept_sel, n_sel[passing])
    # the tracks of a dropped vertex point nowhere (label 0), the others to the renumbered vertex
    offsets = np.concatenate([[0], np.cumsum(n_raw)])
    new_local = np.where(passing, np.cumsum(passing) - 1 - np.repeat(np.concatenate([[0], np.cumsum(n_gv)])[:-1], n_raw), -1)
    n_reset = 0
    for i in range(N_EVENTS):
        raw = ak.to_numpy(events["track_hadidx"][i]); raw_label = ak.to_numpy(events["track_label"][i])
        expected = np.where(raw >= 0, new_local[offsets[i] + np.clip(raw, 0, None)], -1)
        got = ak.to_numpy(tr["track_hadidx"][i]); label = ak.to_numpy(tr["track_label"][i])
        assert np.array_equal(got, expected), i
        reset = (raw >= 0) & (expected < 0)
        n_reset += int(reset.sum())
        assert (label[reset] == 0).all() and np.array_equal(label[~reset], raw_label[~reset])
        # the daughters of the kept vertices only, renumbered; the pair labels follow the new track_hadidx
        d_idx = ak.to_numpy(df["GVDaughters"]["GVDaughters_hadronIndex"][i])
        assert (d_idx < n_gv[i]).all() and np.array_equal(np.bincount(d_idx, minlength=n_gv[i]), ak.to_numpy(gv["GV_nDaughters"][i]))
        i1, i2 = ak.to_numpy(df["trackPair"]["trackPair_trk1Idx"][i]), ak.to_numpy(df["trackPair"]["trackPair_trk2Idx"][i])
        assert np.array_equal(ak.to_numpy(df["trackPair"]["trackPair_edgeLabel"][i]), ((got[i1] == got[i2]) & (got[i1] >= 0)).astype(np.int32))
    assert n_reset > 0
    # GV_nTracks of the kept vertices counts the renumbered tracks
    for i in range(N_EVENTS):
        got = ak.to_numpy(tr["track_hadidx"][i])
        assert ak.to_numpy(gv["GV_nTracks"][i]).tolist() == (np.bincount(got[got >= 0], minlength=n_gv[i]).tolist() if n_gv[i] else [])


def test_tree_without_daughters_is_refused(tmp_path):
    events = synthetic_events()[0]
    path = tmp_path / "nodau.root"
    with uproot.recreate(path) as f:
        f["Events"] = {k: v for k, v in events.items() if not k.startswith("GVDaughters")}
    with pytest.raises(ValueError, match="candidate vertex selection"):
        conv.main([str(path), "--outdir", str(tmp_path)])


def test_norm_par_writes_both_dicts(root_file, tmp_path):
    path = root_file[0]
    conv.main([str(path), "--outdir", str(tmp_path)])
    track_out, vertex_out = tmp_path / "track.yaml", tmp_path / "vertex.yaml"
    assert norm.main(["--input", str(tmp_path / "ttbar_synth.parquet"), "--output", str(track_out), "--vertex-output", str(vertex_out)]) == 0
    track = yaml.safe_load(track_out.read_text())
    vertex = yaml.safe_load(vertex_out.read_text())
    from sv.dataset.features import TRACK_FEATURES, VERTEX_TARGETS
    assert set(track) == set(TRACK_FEATURES) and set(vertex) == set(VERTEX_TARGETS)
    assert track["sinphi"] == {"type": "std", "mean": 0.0, "std": 1.0}
    assert track["eta"]["type"] == "min_max_sym" and track["eta"]["min"] < 0 < track["eta"]["max"]
    assert vertex["pt"]["fn"] == "log" and vertex["lxy"]["fn"] == "log1p" and vertex["lz"]["std"] > 0
    assert "fn" not in track["pt"]  # the log is part of the derivation, not of the dict
