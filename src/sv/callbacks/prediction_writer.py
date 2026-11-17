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
  GV_pred_class           predicted class (0 B, 1 BtoC, 2 C, 3 Other, 4 null)
  GV_pred_prob_<class>    class probabilities (B, BtoC, C, Other, null)
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
  track_seed              dynamic queries only: the track is the seed of a target vertex (the `query_init` target)
  track_seed_prob         dynamic queries only: the `query_init` seed probability (-1 for unselected tracks); the
                          queries are the tracks with seed_prob >= threshold, at most num_queries (the most probable)

at the top level:
  fakeGV                  one entry per predicted vertex not matched to a target: fakeGV_slot, fakeGV_class,
                          fakeGV_prob_<class>, fakeGV_n_tracks, fakeGV_<target> (un-scaled)
  n_fake_vertices, event_number
  slot                    the raw output of every query slot, before any threshold, to scan the working point offline
                          (fake rate vs efficiency, plot_sv_eval.roc_section); one entry per slot:
                            slot_prob_<class>    class probabilities (the vertex score is 1 - slot_prob_null)
                            slot_target_idx      index into GV of the target vertex matched to the slot, -1 if none
                            slot_active          False for the padded slots of a dynamic-query model
                            slot_pred_valid      the slot is a predicted vertex at the model's working point
                            slot_n_tracks, slot_n_correct   tracks in the predicted mask / shared with the truth mask
                                                 at the model's working point (pred_threshold)
                            slot_mask_prob       mask probability of every selected track (in the order of the
                                                 selected tracks of the event), one list per slot
                            slot_truth_mask      truth mask of the matched target on the same tracks (False if none)

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
        self._slot = defaultdict(list)  # per-slot fields, one array per event; (Q, n_sel) for the two mask fields
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
        # (B, Q), dynamic queries only: the slot holds a real query. hepattn puts it in the encoder outputs and in a
        # *copy* of the targets made inside MaskFormer.loss, never in the batch targets seen here
        encoder = outputs.get("encoder", {})
        query_mask = encoder.get("query_mask", targets.get("query_mask"))
        query_mask = query_mask.detach().cpu() if query_mask is not None else None
        # (B, N), dynamic queries only: the `query_init` seed probability of every track and its target
        seed_logit = encoder.get("query_init", {}).get(f"{self.track_collection}_logit")
        seed_prob = seed_logit.detach().float().sigmoid().cpu() if seed_logit is not None else None
        seed_target = targets.get(f"{self.track_collection}_seed")
        seed_target = seed_target.cpu() if seed_target is not None else None

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
            per_track = [("pred_match_idx", match, -1), ("pred_slot", slot, -1), ("pred_prob", np.where(claimed, best_prob, 0.0).astype(np.float32), 0.0)]
            if seed_prob is not None:
                per_track.append(("seed_prob", seed_prob[i, :n_sel].numpy().astype(np.float32), -1.0))
            if seed_target is not None:
                per_track.append(("seed", seed_target[i, :n_sel].numpy().astype(bool), False))
            for name, values, fill in per_track:
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

            # ---- every slot, before any threshold (the offline working-point scan)
            sl = self._slot
            for k, name in enumerate(PROB_NAMES):
                sl[f"prob_{name}"].append(cpf[:, k])
            sl["target_idx"].append(np.where(matched_q, np.arange(len(matched_q)), -1).astype(np.int64))
            sl["active"].append(query_mask[i].numpy().astype(bool) if query_mask is not None else np.ones(len(matched_q), dtype=bool))
            sl["pred_valid"].append(valid_q)
            sl["n_tracks"].append(mask_i.sum(-1).astype(np.int64))
            sl["n_correct"].append((mask_i & truth_i & matched_q[:, None]).sum(-1).astype(np.int64))
            sl["mask_prob"].append(mask_prob[i, :, :n_sel].numpy().astype(np.float32))
            sl["truth_mask"].append(truth_i & matched_q[:, None])

    def _input_events(self, event_numbers: np.ndarray) -> ak.Array:
        print(f"Loading the test file(s) to merge in the predictions: {self.test_path} ({len(self.ds.files)} files)")
        return ak.from_arrow(read_parquet_files(self.ds.files))[event_numbers]

    @staticmethod
    def _to_doubly_jagged(per_event_arrays: list[np.ndarray]) -> ak.Array:
        """A (events, slots, tracks) jagged array from per-event (Q, n) arrays, preserving the dtype."""
        if not per_event_arrays:
            return ak.Array([])
        flat = np.concatenate([a.reshape(-1) for a in per_event_arrays])
        inner = np.concatenate([np.full(a.shape[0], a.shape[1], dtype=np.int64) for a in per_event_arrays])
        outer = np.array([a.shape[0] for a in per_event_arrays], dtype=np.int64)
        return ak.unflatten(ak.unflatten(flat, inner), outer)

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
        slot = ak.zip({f"slot_{name}": (self._to_doubly_jagged(arrays) if arrays and arrays[0].ndim == 2 else self._to_jagged(arrays))
                       for name, arrays in self._slot.items()}, depth_limit=1)
        out = ak.with_field(out, slot, "slot")
        out = ak.with_field(out, ak.Array(np.asarray(self._n_fake, dtype=np.int64)), "n_fake_vertices")
        out = ak.with_field(out, ak.Array(event_numbers), "event_number")
        return out
