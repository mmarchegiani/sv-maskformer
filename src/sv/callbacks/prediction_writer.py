"""Writes the SV MaskFormer test-time predictions next to the checkpoint, as a parquet file with the schema of the
test file (sv.scripts.dataset.convert_root) plus prediction branches (the SV schema of
`maskformer.callbacks.PredictionWriterBase`).

The decoder outputs of every event are permuted by the model's Hungarian matcher so that query slot `i` corresponds
to vertex `i` of the event's `GV` collection for every target vertex (`MaskFormer.loss`); the other slots hold
unmatched queries. A query is a *predicted vertex* when its class prediction is not the null class. Added fields:

on the vertex collection (`GV_*`, one entry per vertex of the file; NaN / False / null class for vertices that were
not targets, see SVDataset.min_tracks_for_valid):
  GV_is_target            the vertex was a target of the model
  GV_n_tracks_selected    its tracks after the dataset's track selection (the truth mask)
  GV_pred_valid           its matched query predicts a vertex
  GV_pred_class           predicted class (0 b, 1 c, 2 other, 3 null)
  GV_pred_prob_<class>    class probabilities (b, c, other, null)
  GV_pred_n_tracks        tracks in the predicted mask of the matched query
  GV_pred_n_correct       tracks shared by the predicted and the truth mask (recall = n_correct / n_tracks_selected,
                          purity = n_correct / pred_n_tracks)
  GV_pred_<target>        regressed targets, un-scaled (pt in GeV, lxy/lz in cm, ...; plus GV_pred_phi from sin/cos)

on the track collection (`track_*`, one entry per track of the file):
  track_selected          the track entered the model (the dataset's track selection)
  track_pred_match_idx    the vertex the track is assigned to: the index into GV of the predicted vertex that claims
                          it (the query with the highest mask probability among those predicting the track), -1 when
                          no predicted vertex claims it, -2 when that query is unmatched (a fake vertex)
  track_pred_slot         the claiming query's slot (-1 none), to recover the tracks of the fake vertices
  track_pred_prob         the mask probability of the claiming query (0 when none)

at the top level:
  fakeGV                  one entry per predicted vertex not matched to a target: fakeGV_slot, fakeGV_class,
                          fakeGV_prob_<class>, fakeGV_n_tracks, fakeGV_<target> (un-scaled)
  n_fake_vertices, event_number

The mean test loss per layer/task/term goes to `<ckpt>__test[_<suff>]_losses.json` (base class).
"""

from __future__ import annotations

from collections import defaultdict

import awkward as ak
import numpy as np
import torch

from maskformer.callbacks.prediction_writer import PredictionWriterBase
from maskformer.utils.parquet import read_parquet_files

from sv.dataset.collections import CLASS_NAMES, RESIDUAL_CLASS

#: class names of the probability branches, null last (hepattn's ObjectClassificationTask)
PROB_NAMES = (*CLASS_NAMES, "null")


class SVPredictionWriter(PredictionWriterBase):
    """Prediction writer of the SV dataset: one prediction per truth vertex (matched query), the track assignment and
    the fake vertices, written back into the test file (see the module doc)."""

    output_description = "vertex predictions"
    #: snappy, so that the torch-free plot chain can read the file in an LCG view whose pyarrow lacks zstd
    parquet_compression = "snappy"

    def regression_transforms(self) -> dict:
        # the regression outputs live in the vertex-fitted space (targets scaled with the vertex dict): invert with it
        return self.ds.vertex_scaler.transforms

    def reset_buffers(self) -> None:
        ds = self.ds
        self.track_collection = ds.track_collection
        self.vertex_collection = ds.vertex_collection
        self._by_event = {int(e): i for i, e in enumerate(np.asarray(ds.event_number).tolist())}
        self._gv = defaultdict(list)  # per-vertex fields, one array per event
        self._track = defaultdict(list)  # per-track fields, one array per event (all tracks of the file)
        self._fake = defaultdict(list)  # per-fake fields, one array per event
        self._n_fake = []

    def collect_batch(self, inputs, targets, outputs, preds, event_number: np.ndarray) -> None:
        final = preds["final"]
        particle_valid = targets["particle_valid"].cpu()  # (B, Q): the slot holds a target vertex
        track_valid = targets["track_valid"].cpu()  # (B, N)
        truth_mask = targets["particle_track_valid"].cpu()  # (B, Q, N)
        pred_class = final["classification"]["pflow_class"].detach().cpu()  # (B, Q)
        pred_valid = final["classification"]["pflow_valid"].detach().cpu()  # (B, Q)
        class_prob = outputs["final"]["classification"]["pflow_class_prob"].detach().float().cpu()  # (B, Q, C+1)
        mask_valid = final["mask"][f"pflow_{self.track_collection}_valid"].detach().cpu()  # (B, Q, N)
        mask_prob = outputs["final"]["mask"][f"pflow_{self.track_collection}_logit"].detach().float().sigmoid().cpu()
        regression = self._unscale_regression_preds(final) if "regression" in final else {}

        for i, event in enumerate(event_number.tolist()):
            row = self._by_event[event]
            ds = self.ds
            n_gv = int(ds.n_vertices[row])
            n_sel = int(track_valid[i].sum())
            n_all = int(ds.n_tracks_all[row])
            ts, te = ds.track_offsets[row], ds.track_offsets[row + 1]
            vs, ve = ds.vertex_offsets[row], ds.vertex_offsets[row + 1]
            assert te - ts == n_sel, f"event {event}: {te - ts} selected tracks in the dataset, {n_sel} in the batch"
            is_target = particle_valid[i, :n_gv].numpy().astype(bool)
            valid_q = pred_valid[i].numpy().astype(bool)  # (Q,)
            matched_q = np.zeros(valid_q.shape, dtype=bool)
            matched_q[:n_gv] = is_target
            mask_i = mask_valid[i, :, :n_sel].numpy().astype(bool)  # (Q, n_sel)
            truth_i = truth_mask[i, :, :n_sel].numpy().astype(bool)

            # ---- per truth vertex
            gv = self._gv
            gv["is_target"].append(is_target)
            gv["n_tracks_selected"].append(np.asarray(ds.vertex_n_tracks[vs:ve], dtype=np.int64))
            pv = valid_q[:n_gv] & is_target
            gv["pred_valid"].append(pv)
            pc = pred_class[i, :n_gv].numpy().astype(np.int64).copy()
            pc[~is_target] = RESIDUAL_CLASS
            gv["pred_class"].append(pc)
            cp = class_prob[i, :n_gv].numpy().astype(np.float32).copy()
            cp[~is_target] = np.nan
            for k, name in enumerate(PROB_NAMES):
                gv[f"pred_prob_{name}"].append(cp[:, k])
            n_pred = mask_i[:n_gv].sum(-1).astype(np.int64)
            n_pred[~is_target] = 0
            gv["pred_n_tracks"].append(n_pred)
            n_correct = (mask_i[:n_gv] & truth_i[:n_gv]).sum(-1).astype(np.int64)
            n_correct[~is_target] = 0
            gv["pred_n_correct"].append(n_correct)
            for field, values in regression.items():
                rv = values[i, :n_gv].numpy().astype(np.float32).copy()
                rv[~is_target] = np.nan
                gv[f"pred_{field}"].append(rv)

            # ---- per track: the predicted vertex claiming it (highest mask probability among the predicting queries)
            prob = mask_prob[i, :, :n_sel].numpy().astype(np.float32).copy()
            prob[~mask_i] = -1.0
            prob[~valid_q] = -1.0
            best = prob.argmax(0) if n_sel else np.zeros(0, dtype=np.int64)
            best_prob = prob.max(0) if n_sel else np.zeros(0, dtype=np.float32)
            claimed = best_prob >= 0
            slot = np.where(claimed, best, -1).astype(np.int64)
            match = np.where(claimed, np.where(matched_q[np.clip(best, 0, None)], best, -2), -1).astype(np.int64)
            local = np.asarray(ds.track_local_idx[ts:te], dtype=np.int64)
            tr = self._track
            selected = np.zeros(n_all, dtype=bool)
            selected[local] = True
            tr["selected"].append(selected)
            for name, values, fill in (("pred_match_idx", match, -1), ("pred_slot", slot, -1), ("pred_prob", np.where(claimed, best_prob, 0.0).astype(np.float32), 0.0)):
                full = np.full(n_all, fill, dtype=values.dtype)
                full[local] = values
                tr[name].append(full)

            # ---- fake vertices: predicted vertices in unmatched slots
            fake_slots = np.flatnonzero(valid_q & ~matched_q)
            fk = self._fake
            fk["slot"].append(fake_slots.astype(np.int64))
            fk["class"].append(pred_class[i].numpy().astype(np.int64)[fake_slots])
            cpf = class_prob[i].numpy().astype(np.float32)
            for k, name in enumerate(PROB_NAMES):
                fk[f"prob_{name}"].append(cpf[fake_slots, k])
            fk["n_tracks"].append(mask_i[fake_slots].sum(-1).astype(np.int64))
            for field, values in regression.items():
                fk[field].append(values[i].numpy().astype(np.float32)[fake_slots])
            self._n_fake.append(int(len(fake_slots)))

    def _input_events(self, event_numbers: np.ndarray) -> ak.Array:
        print(f"Loading the test file(s) to merge in the predictions: {self.test_path} ({len(self.ds.files)} files)")
        return ak.from_arrow(read_parquet_files(self.ds.files))[event_numbers]

    def build_output(self, event_numbers: np.ndarray) -> ak.Array:
        out = self._input_events(event_numbers)
        gv = out[self.vertex_collection]
        for name, arrays in self._gv.items():
            gv = ak.with_field(gv, self._to_jagged(arrays), f"{self.vertex_collection}_{name}")
        if f"{self.vertex_collection}_pred_sinphi" in gv.fields and f"{self.vertex_collection}_pred_cosphi" in gv.fields:
            gv = ak.with_field(gv, np.arctan2(gv[f"{self.vertex_collection}_pred_sinphi"], gv[f"{self.vertex_collection}_pred_cosphi"]),
                               f"{self.vertex_collection}_pred_phi")
        out = ak.with_field(out, gv, self.vertex_collection)
        tracks = out[self.track_collection]
        for name, arrays in self._track.items():
            tracks = ak.with_field(tracks, self._to_jagged(arrays), f"{self.track_collection}_{name}")
        out = ak.with_field(out, tracks, self.track_collection)
        fake = ak.zip({f"fakeGV_{name}": self._to_jagged(arrays) for name, arrays in self._fake.items()}, depth_limit=1)
        out = ak.with_field(out, fake, "fakeGV")
        out = ak.with_field(out, ak.Array(np.asarray(self._n_fake, dtype=np.int64)), "n_fake_vertices")
        out = ak.with_field(out, ak.Array(event_numbers), "event_number")
        return out
