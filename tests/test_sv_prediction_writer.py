"""The `slot` collection of SVPredictionWriter (the raw output of every query slot, for the offline working-point
scan of the plot chain): its content for a hand-made batch and its parquet round trip. Needs torch and lightning.

    PYTHONPATH=src python -m pytest tests/test_sv_prediction_writer.py -q
"""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
ak = pytest.importorskip("awkward")

from sv.callbacks.prediction_writer import SVPredictionWriter  # noqa: E402


def make_writer():
    """Two events: event 10 with two vertices (the second without selected tracks, not a target) and 4 of 5 tracks
    selected, event 11 with one vertex and 3 of 4 tracks selected; 3 query slots, tracks padded to 4."""
    writer = SVPredictionWriter()
    writer.ds = SimpleNamespace(
        track_collection="track", vertex_collection="GV", event_number=np.array([10, 11]), n_vertices=np.array([2, 1]),
        n_tracks_all=np.array([5, 4]), track_offsets=np.array([0, 4, 7]), vertex_offsets=np.array([0, 2, 3]),
        vertex_n_tracks=np.array([2, 0, 2]), track_local_idx=np.array([0, 1, 2, 4, 0, 1, 3]),
    )
    writer.reset_buffers()
    return writer


def make_batch():
    truth = torch.zeros(2, 3, 4, dtype=torch.bool)
    truth[0, 0, :2] = True
    truth[1, 0, 1:3] = True
    targets = {
        "particle_valid": torch.tensor([[True, False, False], [True, False, False]]),
        "track_valid": torch.tensor([[True, True, True, True], [True, True, True, False]]),
        "particle_track_valid": truth,
    }
    # event 10: slot 0 finds its vertex with one extra track, slot 2 is a fake with one track; event 11: slot 0 is null
    prob = torch.full((2, 3, 4), 0.1)
    prob[0, 0, :3] = torch.tensor([0.9, 0.8, 0.6])
    prob[0, 2, 3] = 0.7
    prob[1, 0, 1:3] = torch.tensor([0.4, 0.95])
    class_prob = torch.tensor([[[0.7, 0.1, 0.1, 0.05, 0.05], [0.0, 0.0, 0.0, 0.1, 0.9], [0.1, 0.5, 0.1, 0.1, 0.2]],
                               [[0.1, 0.1, 0.1, 0.1, 0.6], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0]]])
    pred_class = class_prob.argmax(-1)
    outputs = {"final": {"classification": {"pflow_class_prob": class_prob}, "mask": {"pflow_track_logit": torch.logit(prob)}}}
    preds = {"final": {"classification": {"pflow_class": pred_class, "pflow_valid": pred_class < 4}, "mask": {"pflow_track_valid": prob >= 0.5}}}
    return targets, outputs, preds, prob


def test_slot_collection(tmp_path, monkeypatch):
    writer = make_writer()
    targets, outputs, preds, prob = make_batch()
    writer.collect_batch({}, targets, outputs, preds, np.array([10, 11]))
    sl = writer._slot
    assert [a.tolist() for a in sl["target_idx"]] == [[0, -1, -1], [0, -1, -1]]
    assert [a.tolist() for a in sl["pred_valid"]] == [[True, False, True], [False, False, False]]
    assert [a.tolist() for a in sl["n_tracks"]] == [[3, 0, 1], [1, 0, 0]]
    assert [a.tolist() for a in sl["n_correct"]] == [[2, 0, 0], [1, 0, 0]]
    assert all(a.all() for a in sl["active"])
    assert np.allclose(sl["prob_null"][0], [0.05, 0.9, 0.2]) and sl["prob_null"][0].dtype == np.float32
    # the mask fields cover the selected tracks of the event only (4 and 3), for every slot
    assert [a.shape for a in sl["mask_prob"]] == [(3, 4), (3, 3)] == [a.shape for a in sl["truth_mask"]]
    assert np.allclose(sl["mask_prob"][0], prob[0].numpy(), atol=1e-6) and np.allclose(sl["mask_prob"][1], prob[1, :, :3].numpy(), atol=1e-6)
    assert sl["truth_mask"][0].tolist() == [[True, True, False, False], [False] * 4, [False] * 4]
    assert sl["truth_mask"][1].tolist() == [[False, True, True], [False] * 3, [False] * 3]
    # consistent with the per-vertex fields and the fakes
    assert writer._gv["pred_n_tracks"][0].tolist() == [3, 0] and writer._gv["pred_n_correct"][0].tolist() == [2, 0]
    assert writer._fake["slot"][0].tolist() == [2] and writer._n_fake == [1, 0]

    events = ak.Array({"GV": ak.zip({"GV_pt": ak.Array([[1.0, 2.0], [3.0]])}, depth_limit=1),
                       "track": ak.zip({"track_pt": ak.Array([[1.0] * 5, [1.0] * 4])}, depth_limit=1)})
    monkeypatch.setattr(writer, "_input_events", lambda event_numbers: events)
    out = writer.build_output(np.array([10, 11]))
    slot = out["slot"]
    assert ak.num(slot["slot_target_idx"], axis=1).tolist() == [3, 3]
    assert ak.num(slot["slot_mask_prob"], axis=2).tolist() == [[4, 4, 4], [3, 3, 3]]
    assert ak.to_list(slot["slot_truth_mask"][1]) == [[False, True, True], [False] * 3, [False] * 3]
    assert str(ak.type(slot["slot_mask_prob"])).endswith("float32")
    path = tmp_path / "pred.parquet"
    ak.to_parquet(out, path, compression="snappy")
    back = ak.from_parquet(path)
    assert ak.to_list(back["slot"]["slot_n_tracks"]) == [[3, 0, 1], [1, 0, 0]]
    assert np.allclose(ak.to_numpy(ak.flatten(back["slot"]["slot_mask_prob"], axis=None)), np.concatenate([a.reshape(-1) for a in sl["mask_prob"]]))


def test_dynamic_queries_query_mask_and_seeds():
    """hepattn puts the dynamic-query mask and the `query_init` logits in outputs["encoder"] (the batch targets never
    get `query_mask`: MaskFormer.loss adds it to a copy)."""
    writer = make_writer()
    targets, outputs, preds, _ = make_batch()
    seed_prob = torch.tensor([[0.9, 0.2, 0.6, 0.1], [0.3, 0.8, 0.1, 0.5]])
    targets["track_seed"] = torch.tensor([[True, False, False, False], [False, True, False, False]])
    outputs["encoder"] = {"query_mask": torch.tensor([[True, True, False], [True, False, False]]),
                          "query_init": {"track_logit": torch.logit(seed_prob)}}
    writer.collect_batch({}, targets, outputs, preds, np.array([10, 11]))
    assert [a.tolist() for a in writer._slot["active"]] == [[True, True, False], [True, False, False]]
    # per track of the file (5 and 4), the unselected tracks filled with -1 / False
    sp = writer._track["seed_prob"]
    assert np.allclose(sp[0], [0.9, 0.2, 0.6, -1.0, 0.1], atol=1e-6) and np.allclose(sp[1], [0.3, 0.8, -1.0, 0.1], atol=1e-6)
    assert [a.tolist() for a in writer._track["seed"]] == [[True, False, False, False, False], [False, True, False, False]]


def test_no_seed_fields_without_dynamic_queries():
    writer = make_writer()
    targets, outputs, preds, _ = make_batch()
    writer.collect_batch({}, targets, outputs, preds, np.array([10, 11]))
    assert "seed_prob" not in writer._track and "seed" not in writer._track
