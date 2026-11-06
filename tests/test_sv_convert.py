"""sv.scripts.dataset.convert_root and compute_norm_par on a synthetic BTV ntuple (written with uproot; torch-free):
the parquet layout, the generated primary vertex, the derived truth fields, the split and the fitted scale dicts.

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
from sv.dataset.collections import PV_FIELDS, TRACK_BRANCHES, VERTEX_BRANCHES, VERTEX_DERIVED  # noqa: E402
from sv.scripts.dataset import compute_norm_par as norm  # noqa: E402
from sv.scripts.dataset import convert_root as conv  # noqa: E402

N_EVENTS = 12


def synthetic_events(seed=0):
    """A small Events tree: per event 2-4 vertices (the first prompt), 20-40 tracks, 30% of them pointing to a vertex."""
    rng = np.random.default_rng(seed)
    events = {}
    n_gv = rng.integers(0, 5, N_EVENTS)
    n_tr = rng.integers(20, 41, N_EVENTS)
    scalars = {"run": np.ones(N_EVENTS, np.uint32), "luminosityBlock": np.arange(N_EVENTS, dtype=np.uint32), "event": np.arange(N_EVENTS, dtype=np.uint64)}
    pv = rng.normal([0.03, -0.02, 0.0], [0.001, 0.001, 3.0], (N_EVENTS, 3))
    gv = {name: [] for name in VERTEX_BRANCHES}
    tr = {name: [] for name in TRACK_BRANCHES}
    gen = {"GenPart_status": [], "GenPart_vx": [], "GenPart_vy": [], "GenPart_vz": []}
    for i in range(N_EVENTS):
        k, n = int(n_gv[i]), int(n_tr[i])
        is_b = (rng.random(k) < 0.4).astype(np.int32)
        is_d = 1 - is_b
        from_hf = ((rng.random(k) < 0.5) & (is_d == 1)).astype(np.int32)
        pos = pv[i] + rng.normal(0, [0.3, 0.3, 0.5], (k, 3))
        gv["GV_Hadron_pdgId"].append(np.where(is_b, 511, 421).astype(np.int32))
        gv["GV_isB"].append(is_b); gv["GV_isD"].append(is_d); gv["GV_fromHF"].append(from_hf)
        gv["GV_toHF"].append(np.zeros(k, np.int32)); gv["GV_pdgClass"].append(np.where(is_b, 511, 421).astype(np.int32))
        gv["GV_nDaughters"].append(rng.integers(2, 5, k).astype(np.int32))
        gv["GV_pt"].append(rng.uniform(10, 100, k).astype(np.float32)); gv["GV_eta"].append(rng.uniform(-2.4, 2.4, k).astype(np.float32))
        gv["GV_phi"].append(rng.uniform(-np.pi, np.pi, k).astype(np.float32))
        for j, a in enumerate("xyz"):
            gv[f"GV_{a}"].append(pos[:, j].astype(np.float32))
            gv[f"GV_{a}_i"].append(np.full(k, pv[i][j], np.float32))
        gv["GV_maxDaughterPairDeltaR"].append(rng.uniform(0, 0.5, k).astype(np.float32))
        hadidx = np.where(rng.random(n) < 0.3, rng.integers(0, max(k, 1), n), -1).astype(np.int32) if k else np.full(n, -1, np.int32)
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
    for d in (gv, tr, gen):
        for name, arrays in d.items():
            events[name] = ak.Array(arrays)
    return events, pv, n_gv, n_tr


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
    cls = ak.to_numpy(ak.flatten(gv["GV_class"])); is_b = ak.to_numpy(ak.flatten(gv["GV_isB"]))
    assert np.all(cls[is_b == 1] == 0) and np.all(cls[is_b == 0] == 1)


def test_convert_split_and_optional_collections(root_file, tmp_path):
    path = root_file[0]
    assert conv.main([str(path), "--outdir", str(tmp_path), "--split", "0.5", "0.25", "0.25", "--max-events", "8"]) == 0
    counts = {name: pq.read_metadata(tmp_path / name / "ttbar_synth.parquet").num_rows for name in ("train", "val", "test")}
    assert counts == {"train": 4, "val": 2, "test": 2}
    with pytest.raises(ValueError):
        conv.split_counts(10, [0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="lacks the branches"):
        conv.main([str(path), "--outdir", str(tmp_path / "x"), "--keep", "GVDaughters"])


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
