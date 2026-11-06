"""sv.dataset.dataset_sv (needs torch, hepattn's FeatureScaler): the tensors of an event, the mask target from
track_hadidx, the target flag, the track selection, the data module mirror and the prediction writer's bookkeeping,
on a synthetic parquet file (the generator of tests/test_sv_convert.py).

    PYTHONPATH=src python -m pytest tests/test_sv_dataset.py -q      # container
"""

import inspect
import pathlib
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hepattn")
ak = pytest.importorskip("awkward")
uproot = pytest.importorskip("uproot")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
from sv.dataset.collections import RESIDUAL_CLASS  # noqa: E402
from sv.dataset.dataset_sv import SVDataModule, SVDataset  # noqa: E402
from sv.scripts.dataset import compute_norm_par as norm  # noqa: E402
from sv.scripts.dataset import convert_root as conv  # noqa: E402
from test_sv_convert import synthetic_events  # noqa: E402

INPUTS = {"track": ["pt", "eta", "sinphi", "cosphi", "ip2d", "ip3dsig", "dz", "charge", "nhits", "passseed"]}
TARGETS = {"vertex": ["pt", "eta", "sinphi", "cosphi", "lxy", "lz"]}


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    """A converted synthetic file and its two scale dicts."""
    events, pv, n_gv, n_tr = synthetic_events(seed=1)
    root = tmp_path_factory.mktemp("svds") / "synth.root"
    with uproot.recreate(root) as f:
        f["Events"] = events
    outdir = root.parent
    conv.main([str(root), "--outdir", str(outdir)])
    parquet = outdir / "synth.parquet"
    track_dict, vertex_dict = outdir / "track.yaml", outdir / "vertex.yaml"
    norm.main(["--input", str(parquet), "--output", str(track_dict), "--vertex-output", str(vertex_dict)])
    return parquet, str(track_dict), str(vertex_dict), n_gv, n_tr


def make(sample, **kwargs):
    parquet, track_dict, vertex_dict, _, _ = sample
    return SVDataset(str(parquet), inputs=INPUTS, targets=TARGETS, scale_dict_path=track_dict, vertex_scale_dict_path=vertex_dict,
                     num_objects=8, max_tracks=64, **kwargs)


def test_event_tensors(sample):
    ds = make(sample)
    parquet, _, _, n_gv, n_tr = sample
    assert len(ds) == len(n_gv)
    df = ak.from_arrow(__import__("pyarrow.parquet").parquet.read_table(parquet))
    for i in range(len(ds)):
        inputs, targets = ds[i]
        n = int(n_tr[i]); k = int(n_gv[i])
        assert inputs["track_features"].shape == (64, len(INPUTS["track"])) and inputs["track_valid"].sum() == n
        assert targets["track_valid"].equal(inputs["track_valid"])
        assert torch.isfinite(inputs["track_features"][:n]).all() and (inputs["track_features"][n:] == 0).all()
        assert targets["particle_valid"].shape == (8,) and targets["particle_track_valid"].shape == (8, 64)
        hadidx = ak.to_numpy(df["track"]["track_hadidx"][i])
        mask = targets["particle_track_valid"].numpy()
        for j in range(k):
            expected = np.zeros(64, dtype=bool)
            expected[np.flatnonzero(hadidx == j)] = True
            if targets["particle_valid"][j]:
                assert np.array_equal(mask[j], expected), (i, j)
                assert targets["particle_class"][j] in (0, 1)
            else:  # not a target: no tracks, null class
                assert not mask[j].any() and targets["particle_class"][j] == RESIDUAL_CLASS
        assert (targets["particle_class"][k:] == RESIDUAL_CLASS).all() and not targets["particle_valid"][k:].any()
        for name in TARGETS["vertex"]:
            v = targets[f"particle_{name}"]
            assert torch.isfinite(v[:k]).all() and torch.isnan(v[k:]).all()
        # raw kinematics for the positional encoding
        assert torch.allclose(inputs["track_pt"][:n], torch.as_tensor(ak.to_numpy(df["track"]["track_pt"][i])).float())
        assert int(targets["event_number"]) == i
    # the regression targets are the scaled derived quantities: un-scaling lxy gives the file's GV_lxy
    inputs, targets = ds[0]
    k = int(n_gv[0])
    if k:
        lxy = ds.vertex_scaler.transforms["lxy"].inverse_transform(targets["particle_lxy"][:k])
        assert np.allclose(lxy.numpy(), ak.to_numpy(df["GV"]["GV_lxy"][0]), atol=1e-4)


def test_min_tracks_for_valid_and_selection(sample):
    all_tracks = make(sample)
    strict = make(sample, min_tracks_for_valid=2)
    assert int(strict.vertex_is_target.sum()) <= int(all_tracks.vertex_is_target.sum())
    assert np.array_equal(strict.vertex_is_target.numpy(), strict.vertex_n_tracks >= 2)
    assert np.array_equal(all_tracks.vertex_is_target.numpy(), all_tracks.vertex_n_tracks >= 1)
    sel = make(sample, track_require_pass_seed=True, track_max_per_event=10, track_min_pt=1.0)
    assert sel.n_tracks.max() <= 10 and (sel.track_pt > 1.0).all()
    # the original track index survives the selection
    parquet = sample[0]
    df = ak.from_arrow(__import__("pyarrow.parquet").parquet.read_table(parquet))
    ts, te = sel.track_offsets[0], sel.track_offsets[1]
    pt_file = ak.to_numpy(df["track"]["track_pt"][0])
    assert np.allclose(sel.track_pt[ts:te].numpy(), pt_file[sel.track_local_idx[ts:te]])
    assert sel.counts["track_selection"]["tracks_kept"] < sel.counts["track_selection"]["tracks_loaded"]


def test_unknown_inputs_and_missing_dict_entries(sample):
    parquet, track_dict, vertex_dict, _, _ = sample
    with pytest.raises(ValueError, match="unknown track features"):
        SVDataset(str(parquet), inputs={"track": ["pt", "jetpt"]}, targets=TARGETS, scale_dict_path=track_dict, vertex_scale_dict_path=vertex_dict)
    with pytest.raises(ValueError, match="unknown vertex targets"):
        SVDataset(str(parquet), inputs=INPUTS, targets={"vertex": ["mass"]}, scale_dict_path=track_dict, vertex_scale_dict_path=vertex_dict)
    with pytest.raises(KeyError, match="no entry"):
        SVDataset(str(parquet), inputs=INPUTS, targets=TARGETS, scale_dict_path=vertex_dict, vertex_scale_dict_path=vertex_dict)


def test_candidate_vertex_selection_is_checked(sample, tmp_path, capsys):
    """A file whose vertices fail the converter's candidate selection is refused; one converted before the selection
    existed (no GV_nSelectedDaughters) is accepted with a note."""
    pq = __import__("pyarrow.parquet").parquet
    parquet, track_dict, vertex_dict, _, _ = sample
    df = ak.from_arrow(pq.read_table(parquet))
    gv = df["GV"]
    gv = ak.with_field(gv, ak.where(ak.local_index(gv["GV_pt"], axis=1) == 0, 5.0, gv["GV_pt"]), "GV_pt")  # the first vertex of every event: pT 5
    bad = tmp_path / "bad.parquet"
    pq.write_table(ak.to_arrow_table(ak.with_field(df, gv, "GV"), extensionarray=False), bad)
    with pytest.raises(ValueError, match="fail the candidate vertex selection"):
        SVDataset(str(bad), inputs=INPUTS, targets=TARGETS, scale_dict_path=track_dict, vertex_scale_dict_path=vertex_dict)
    old = tmp_path / "old.parquet"
    pq.write_table(ak.to_arrow_table(ak.with_field(df, ak.without_field(df["GV"], "GV_nSelectedDaughters"), "GV"), extensionarray=False), old)
    ds = SVDataset(str(old), inputs=INPUTS, targets=TARGETS, scale_dict_path=track_dict, vertex_scale_dict_path=vertex_dict)
    assert len(ds) == len(df) and "no GV_nSelectedDaughters" in capsys.readouterr().out


def test_datamodule_mirrors_the_dataset_arguments():
    module_params = inspect.signature(SVDataModule.__init__).parameters
    for name, param in inspect.signature(SVDataset.__init__).parameters.items():
        if name in {"self", "filepath", "num_events"}:
            continue
        assert name in module_params, f"SVDataModule.__init__ lacks SVDataset's {name}"
        assert module_params[name].default == param.default, name


def test_datamodule_batches(sample):
    parquet, track_dict, vertex_dict, _, _ = sample
    dm = SVDataModule(batch_size=4, num_workers=0, num_train=-1, num_val=-1, num_test=-1, scale_dict_path=track_dict,
                      vertex_scale_dict_path=vertex_dict, inputs=INPUTS, targets=TARGETS, train_path=str(parquet), valid_path=str(parquet),
                      test_path=str(parquet), num_objects=8, max_tracks=64, pin_memory=False)

    class FakeTrainer:
        is_global_zero = True
        log_dir = None

    dm.trainer = FakeTrainer()
    dm.setup("fit")
    inputs, targets = next(iter(dm.train_dataloader()))
    assert inputs["track_features"].shape == (4, 64, len(INPUTS["track"])) and targets["particle_track_valid"].shape == (4, 8, 64)
    assert targets["particle_class"].dtype == torch.int64 and targets["particle_lxy"].shape == (4, 8)
    from maskformer.utils.finite_checks import assert_batch_finite
    assert_batch_finite(inputs, targets, constituent="track")
