"""The SV dataset: reconstructed tracks of an event as the input constituents, heavy-flavour decay vertices as the
objects to find, assign tracks to, classify and fit (sv.dataset.collections for the collections and the truth).

`SVDataset` reads the parquet files written by `maskformer-sv convert`, derives and scales the track input features
(sv.dataset.features.TRACK_FEATURES, the track scale dict) and the vertex regression targets (VERTEX_TARGETS, the
vertex scale dict), applies an optional track selection, and returns per event the padded tensors hepattn's
MaskFormer and the repository's `MaskformerWrapper` expect. The object is called `particle` in the tensors (hepattn's
target_object, shared with the HGCAL pipeline), the constituent `track`:

    inputs
      track_features      (max_tracks, F)  the scaled input features, in the order of `inputs.track`
      track_valid         (max_tracks,)    padding mask
      track_pt/eta/phi    (max_tracks,)    raw kinematics (for the positional encoding and the prediction writer)
    targets
      particle_valid        (num_objects,)             the vertex is a target (see min_tracks_for_valid)
      particle_class        (num_objects,)             0 b, 1 c, 2 other; RESIDUAL_CLASS for non-targets and padding
      particle_track_valid  (num_objects, max_tracks)  the mask target: the tracks of each target vertex (track_hadidx)
      track_valid           (max_tracks,)              the padding mask again (hepattn's mask task reads it here)
      particle_<target>     (num_objects,)             scaled regression targets, NaN beyond the event's vertices
      event_number          ()                         row of the event in the concatenated test file(s)

Events are kept when they have at least one selected track, at most `max_tracks` of them and at most `num_objects`
vertices (none of the 5000 ttbar events is lost with the defaults). An event without any vertex is a valid training
event (every query must predict the null class). A vertex with fewer than `min_tracks_for_valid` selected tracks is
not a target: its class is the null class, its tracks are background, and the prediction writer reports it as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import awkward as ak
import lightning as L
import numpy as np
import pyarrow.parquet as pq
import torch
from hepattn.utils.scaling import FeatureScaler
from torch.utils.data import DataLoader, Dataset

from maskformer.utils.finite_checks import assert_finite, assert_scaler_finite, assert_transform_domain
from maskformer.utils.parquet import list_parquet_files, read_parquet_files, split_parquet_files
from maskformer.utils.paths import resolve_scale_dict_path

from sv.dataset.collections import (
    NUM_CLASSES,
    PV_FIELDS,
    RESIDUAL_CLASS,
    TRACK_COLLECTION,
    TRACK_TRUTH_IDX,
    VERTEX_COLLECTION,
    vertex_category,
    vertex_class,
)
from sv.dataset.features import PV_KEYS, TRACK_FEATURES, VERTEX_TARGETS, branches_needed, derive

#: raw track branches always loaded (kinematics for the positional encoding / writer, the truth link, the selections)
RAW_TRACK_BRANCHES = ("track_pt", "track_eta", "track_phi", "track_passSeed", "track_isHighPurity", TRACK_TRUTH_IDX)
#: raw vertex branches always loaded (classes, counts)
RAW_VERTEX_BRANCHES = ("GV_pt", "GV_isB", "GV_isD", "GV_fromHF")


def make_feature_scaler(scale_dict_path: str, names, strict: bool, what: str) -> FeatureScaler:
    """hepattn FeatureScaler of `scale_dict_path`; with `strict`, every name in `names` must have an entry (hepattn
    silently falls back to the identity for a missing variable)."""
    scaler = FeatureScaler(scale_dict_path)
    missing = [n for n in names if n not in scaler.scale_dict]
    if missing and strict:
        raise KeyError(f"{what} scale dict {scale_dict_path} has no entry for {missing} (has: {sorted(scaler.scale_dict)})")
    if missing:
        print(f"WARNING: {what} scale dict {scale_dict_path} has no entry for {missing}: identity transform")
    assert_scaler_finite(scaler, scale_dict_path)
    return scaler


class SVDataset(Dataset):
    def __init__(
        self,
        filepath: str | list[str],
        inputs: dict,
        targets: dict,
        scale_dict_path: str,
        vertex_scale_dict_path: str,
        track_collection: str = TRACK_COLLECTION,
        vertex_collection: str = VERTEX_COLLECTION,
        num_events: int = -1,
        num_objects: int = 16,
        max_tracks: int = 1024,
        min_tracks_for_valid: int = 1,
        track_min_pt: float = 0.0,
        track_max_abs_eta: float | None = None,
        track_require_pass_seed: bool = False,
        track_require_high_purity: bool = False,
        track_max_per_event: int | None = None,
        strict_scaling: bool = True,
        verbose: bool = False,
    ):
        """Args:
        filepath: a parquet file, a directory of them or a list of files (read as one table, in order).
        inputs: `{"track": [feature names]}`, the input features (sv.dataset.features.TRACK_FEATURES), in order.
        targets: `{"vertex": [target names]}`, the regression targets (VERTEX_TARGETS); each becomes `particle_<name>`.
        scale_dict_path: track scale dict (params/sv_track_var_transform.yaml), fitted on the derived track features.
        vertex_scale_dict_path: vertex scale dict (params/sv_vertex_var_transform.yaml), fitted on the derived targets;
            the regression task and the prediction writer must be given the same file.
        track_collection, vertex_collection: the struct columns of the parquet file.
        num_events: use only the first N events (-1: all).
        num_objects: object slots (= decoder queries); events with more vertices are dropped.
        max_tracks: padded number of tracks; events with more selected tracks are dropped.
        min_tracks_for_valid: a vertex is a target only with at least this many selected tracks pointing to it.
        track_min_pt, track_max_abs_eta, track_require_pass_seed, track_require_high_purity, track_max_per_event:
            the track selection (all off by default); `track_max_per_event` keeps the highest-pT tracks.
        strict_scaling: a feature/target missing from its scale dict is an error (else identity with a warning).
        """
        super().__init__()
        self.filepath = filepath if isinstance(filepath, (str, Path)) else f"{len(filepath)} parquet files"
        self.inputs = inputs
        self.targets = targets
        self.track_collection = track_collection
        self.vertex_collection = vertex_collection
        self.num_objects = int(num_objects)
        self.max_tracks = int(max_tracks)
        self.min_tracks_for_valid = int(min_tracks_for_valid)
        self.track_min_pt = float(track_min_pt)
        self.track_max_abs_eta = None if track_max_abs_eta is None else float(track_max_abs_eta)
        self.track_require_pass_seed = bool(track_require_pass_seed)
        self.track_require_high_purity = bool(track_require_high_purity)
        self.track_max_per_event = None if track_max_per_event is None else int(track_max_per_event)
        self.verbose = verbose

        self.feature_names = [str(f) for f in inputs.get("track", [])]
        self.target_names = [str(t) for t in targets.get("vertex", [])]
        if not self.feature_names:
            raise ValueError("inputs.track lists no feature; known: " + ", ".join(TRACK_FEATURES))
        unknown = [f for f in self.feature_names if f not in TRACK_FEATURES]
        if unknown:
            raise ValueError(f"unknown track features {unknown}; known: {sorted(TRACK_FEATURES)}")
        unknown = [t for t in self.target_names if t not in VERTEX_TARGETS]
        if unknown:
            raise ValueError(f"unknown vertex targets {unknown}; known: {sorted(VERTEX_TARGETS)}")

        # Two scale dicts, one per collection: the track one scales the input features, the vertex one the regression
        # targets (and, through the config alias, the regression task's outputs and the prediction writer).
        self.scale_dict_path = resolve_scale_dict_path(scale_dict_path)
        self.vertex_scale_dict_path = resolve_scale_dict_path(vertex_scale_dict_path)
        self.scaler = make_feature_scaler(self.scale_dict_path, self.feature_names, strict_scaling, "track")
        self.vertex_scaler = make_feature_scaler(self.vertex_scale_dict_path, self.target_names, strict_scaling, "vertex")

        # Footers only: per-file event counts, so that num_events > 0 opens only the files it needs
        candidate_files = list_parquet_files(filepath)
        n_rows = np.cumsum([pq.read_metadata(f).num_rows for f in candidate_files])
        n_used = len(candidate_files)
        if num_events > 0:
            n_used = min(n_used, int(np.searchsorted(n_rows, num_events)) + 1)
        self.files = candidate_files[:n_used]
        print(f"Loading SV dataset from {self.filepath} with {num_events} events "
              f"({len(self.files)}/{len(candidate_files)} files, {int(n_rows[n_used - 1]):,} events in them)")
        self.load_data(num_events)

    # ---- loading ----------------------------------------------------------------------------------------------

    def _read(self):
        columns = [self.track_collection, self.vertex_collection, *PV_FIELDS]
        available = set(pq.read_metadata(self.files[0]).schema.to_arrow_schema().names)
        missing = [c for c in columns if c not in available and c not in PV_FIELDS]
        if missing:
            raise ValueError(f"{self.files[0]} lacks the columns {missing} (has: {sorted(available)})")
        self.has_pv = all(c in available for c in PV_FIELDS)
        if not self.has_pv:
            print(f"NOTE: {self.filepath} carries no {PV_FIELDS}: the flight distances are measured from the origin")
            columns = [c for c in columns if c not in PV_FIELDS]
        return ak.from_arrow(read_parquet_files(self.files, columns=columns))

    def load_data(self, num_events: int):
        df = self._read()
        if num_events > 0:
            df = df[:num_events]
        n_loaded = len(df)
        tracks, vertices = df[self.track_collection], df[self.vertex_collection]

        # ---- tracks: flat tables of the branches the features and the selection need
        track_branches = list(dict.fromkeys([*RAW_TRACK_BRANCHES, *branches_needed(TRACK_FEATURES, self.feature_names)]))
        missing = [b for b in track_branches if b not in tracks.fields]
        if missing:
            raise ValueError(f"'{self.track_collection}' in {self.filepath} has no branch(es) {missing}")
        n_tracks_all = ak.to_numpy(ak.num(tracks["track_pt"], axis=1)).astype(np.int64)
        track_table = {b: ak.to_numpy(ak.flatten(tracks[b])) for b in track_branches}
        track_event = np.repeat(np.arange(n_loaded), n_tracks_all)
        local_idx = ak.to_numpy(ak.flatten(ak.local_index(tracks["track_pt"], axis=1))).astype(np.int64)
        for b, values in track_table.items():
            assert_finite(f"branch '{b}'", values, context=str(self.filepath))

        keep = self._track_selection(track_table, track_event, n_loaded)
        n_selected = (keep.sum(), len(keep))
        track_table = {b: v[keep] for b, v in track_table.items()}
        track_event, local_idx = track_event[keep], local_idx[keep]
        n_tracks = np.bincount(track_event, minlength=n_loaded).astype(np.int64)

        # ---- vertices
        vertex_branches = list(dict.fromkeys([*RAW_VERTEX_BRANCHES, *[b for b in branches_needed(VERTEX_TARGETS, self.target_names) if b not in PV_KEYS]]))
        missing = [b for b in vertex_branches if b not in vertices.fields]
        if missing:
            raise ValueError(f"'{self.vertex_collection}' in {self.filepath} has no branch(es) {missing}")
        n_vertices = ak.to_numpy(ak.num(vertices["GV_pt"], axis=1)).astype(np.int64)
        vertex_table = {b: ak.to_numpy(ak.flatten(vertices[b])) for b in vertex_branches}
        for axis, key in zip("xyz", PV_KEYS):
            pv = ak.to_numpy(df[f"PV_{axis}"]).astype(np.float64) if self.has_pv else np.zeros(n_loaded)
            vertex_table[key] = np.repeat(pv, n_vertices)
        for b, values in vertex_table.items():
            assert_finite(f"branch '{b}'", values, context=str(self.filepath))
        vertex_offsets = np.concatenate([[0], np.cumsum(n_vertices)])

        # truth link of the selected tracks: event-local vertex index -> global vertex row
        hadidx = track_table[TRACK_TRUTH_IDX].astype(np.int64)
        linked = hadidx >= 0
        if linked.any() and (hadidx[linked] >= n_vertices[track_event[linked]]).any():
            raise ValueError(f"{TRACK_TRUTH_IDX} points outside the '{self.vertex_collection}' collection in {self.filepath}")
        n_tracks_per_vertex = np.bincount(vertex_offsets[track_event[linked]] + hadidx[linked], minlength=int(n_vertices.sum()))
        is_target = n_tracks_per_vertex >= self.min_tracks_for_valid

        # ---- event selection and cutflow
        selections = {
            "n_tracks > 0": n_tracks > 0,
            f"n_tracks <= max_tracks ({self.max_tracks})": n_tracks <= self.max_tracks,
            f"n_vertices <= num_objects ({self.num_objects})": n_vertices <= self.num_objects,
        }
        passing = np.ones(n_loaded, dtype=bool)
        cutflow = {"loaded": int(n_loaded)}
        for name, sel in selections.items():
            passing &= sel
            cutflow[name] = int(passing.sum())
        print(f"Track selection keeps {n_selected[0]:,}/{n_selected[1]:,} tracks; event cutflow: {cutflow}")
        keep_track = passing[track_event]
        keep_vertex = passing[np.repeat(np.arange(n_loaded), n_vertices)]

        def multiplicity(n):
            n = np.asarray(n)
            return {"mean": float(n.mean()) if len(n) else 0.0, "std": float(n.std()) if len(n) else 0.0,
                    "min": int(n.min()) if len(n) else 0, "max": int(n.max()) if len(n) else 0}

        self.counts = {
            "filepath": str(self.filepath),
            "files": [str(f) for f in self.files],
            "track_collection": self.track_collection,
            "vertex_collection": self.vertex_collection,
            "track_selection": {
                "min_pt": self.track_min_pt, "max_abs_eta": self.track_max_abs_eta, "require_pass_seed": self.track_require_pass_seed,
                "require_high_purity": self.track_require_high_purity, "max_per_event": self.track_max_per_event,
                "tracks_kept": int(n_selected[0]), "tracks_loaded": int(n_selected[1]),
            },
            "min_tracks_for_valid": self.min_tracks_for_valid,
            "cutflow": cutflow,
            "selected": {
                "events": int(passing.sum()),
                "tracks": int(n_tracks[passing].sum()),
                "vertices": int(n_vertices[passing].sum()),
                "target_vertices": int(is_target[keep_vertex].sum()),
                "events_without_vertex": int((n_vertices[passing] == 0).sum()),
            },
            "track_counts": {"loaded": multiplicity(n_tracks_all), "selected": multiplicity(n_tracks[passing])},
            "vertex_counts": {"loaded": multiplicity(n_vertices), "selected": multiplicity(n_vertices[passing])},
            "tracks_per_vertex": multiplicity(n_tracks_per_vertex[keep_vertex]),
        }

        # ---- derived, scaled features (flat over the surviving events)
        track_table = {b: v[keep_track] for b, v in track_table.items()}
        derived = derive(TRACK_FEATURES, self.feature_names, track_table)
        columns = []
        for name in self.feature_names:
            raw = torch.as_tensor(derived[name], dtype=torch.float32)
            transform = self.scaler.transforms[name]
            assert_transform_domain(f"track feature '{name}'", raw, transform, context=str(self.filepath))
            scaled = transform.transform(raw)
            assert_finite(f"scaled track feature '{name}'", scaled, context=str(self.filepath))
            columns.append(scaled)
        self.track_features = torch.stack(columns, dim=-1) if columns else torch.zeros(0, 0)
        self.track_pt = torch.as_tensor(track_table["track_pt"], dtype=torch.float32)
        self.track_eta = torch.as_tensor(track_table["track_eta"], dtype=torch.float32)
        self.track_phi = torch.as_tensor(track_table["track_phi"], dtype=torch.float32)
        self.track_hadidx = torch.as_tensor(track_table[TRACK_TRUTH_IDX], dtype=torch.int64)
        self.track_local_idx = local_idx[keep_track]

        vertex_table = {b: v[keep_vertex] for b, v in vertex_table.items()}
        derived = derive(VERTEX_TARGETS, self.target_names, vertex_table)
        self.vertex_targets = {}
        for name in self.target_names:
            raw = torch.as_tensor(derived[name], dtype=torch.float32)
            transform = self.vertex_scaler.transforms[name]
            assert_transform_domain(f"vertex target '{name}'", raw, transform, context=str(self.filepath))
            scaled = transform.transform(raw)
            assert_finite(f"scaled vertex target '{name}'", scaled, context=str(self.filepath))
            self.vertex_targets[name] = scaled
        self.vertex_class = torch.as_tensor(vertex_class(vertex_table["GV_isB"], vertex_table["GV_isD"]), dtype=torch.int64)
        self.vertex_category = vertex_category(vertex_table["GV_isB"], vertex_table["GV_isD"], vertex_table["GV_fromHF"])
        self.vertex_is_target = torch.as_tensor(is_target[keep_vertex], dtype=torch.bool)
        self.vertex_n_tracks = n_tracks_per_vertex[keep_vertex].astype(np.int64)

        # ---- per-event bookkeeping of the surviving events
        self.event_number = np.flatnonzero(passing).astype(np.int64)
        self.n_tracks = n_tracks[passing]
        self.n_tracks_all = n_tracks_all[passing]  # tracks in the file, before the selection
        self.n_vertices = n_vertices[passing]
        self.track_offsets = np.concatenate([[0], np.cumsum(self.n_tracks)])
        self.vertex_offsets = np.concatenate([[0], np.cumsum(self.n_vertices)])
        self.num_events = int(passing.sum())
        print(f"SV dataset: {self.num_events:,} events, {int(self.n_tracks.sum()):,} tracks "
              f"(mean {self.n_tracks.mean():.0f}/event), {int(self.n_vertices.sum()):,} vertices "
              f"({int(self.vertex_is_target.sum()):,} targets with >= {self.min_tracks_for_valid} track(s); "
              f"{self.counts['selected']['events_without_vertex']} events without a vertex)")

    def _track_selection(self, table: dict, track_event: np.ndarray, n_events: int) -> np.ndarray:
        keep = np.ones(len(track_event), dtype=bool)
        if self.track_min_pt > 0:
            keep &= table["track_pt"] > self.track_min_pt
        if self.track_max_abs_eta is not None:
            keep &= np.abs(table["track_eta"]) < self.track_max_abs_eta
        if self.track_require_pass_seed:
            keep &= table["track_passSeed"].astype(bool)
        if self.track_require_high_purity:
            keep &= table["track_isHighPurity"].astype(bool)
        if self.track_max_per_event is not None:
            # rank of each track by decreasing pT within its event, among the tracks kept so far
            pt = np.where(keep, table["track_pt"], -np.inf)
            order = np.lexsort((-pt, track_event))
            rank = np.empty(len(pt), dtype=np.int64)
            starts = np.concatenate([[0], np.cumsum(np.bincount(track_event, minlength=n_events))])
            rank[order] = np.arange(len(pt)) - starts[track_event[order]]
            keep &= rank < self.track_max_per_event
        return keep

    # ---- items ------------------------------------------------------------------------------------------------

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        ts, te = self.track_offsets[idx], self.track_offsets[idx + 1]
        vs, ve = self.vertex_offsets[idx], self.vertex_offsets[idx + 1]
        n_tracks, n_vertices = te - ts, ve - vs

        features = torch.zeros(self.max_tracks, self.track_features.shape[1], dtype=torch.float32)
        features[:n_tracks] = self.track_features[ts:te]
        track_valid = torch.zeros(self.max_tracks, dtype=torch.bool)
        track_valid[:n_tracks] = True

        def pad(values):
            out = torch.zeros(self.max_tracks, dtype=torch.float32)
            out[:n_tracks] = values[ts:te]
            return out

        inputs = {
            "track_features": features,
            "track_valid": track_valid,
            "track_pt": pad(self.track_pt),
            "track_eta": pad(self.track_eta),
            "track_phi": pad(self.track_phi),
        }

        is_target = self.vertex_is_target[vs:ve]
        particle_valid = torch.zeros(self.num_objects, dtype=torch.bool)
        particle_valid[:n_vertices] = is_target
        particle_class = torch.full((self.num_objects,), RESIDUAL_CLASS, dtype=torch.int64)
        particle_class[:n_vertices] = torch.where(is_target, self.vertex_class[vs:ve], torch.full_like(is_target, RESIDUAL_CLASS, dtype=torch.int64))

        mask = torch.zeros(self.num_objects, self.max_tracks, dtype=torch.bool)
        hadidx = self.track_hadidx[ts:te]
        linked = hadidx >= 0
        mask[hadidx[linked], torch.arange(n_tracks)[linked]] = True
        mask[:n_vertices][~is_target] = False  # the tracks of a non-target vertex are background

        targets = {
            "particle_valid": particle_valid,
            "particle_class": particle_class,
            "particle_track_valid": mask,
            "track_valid": track_valid,
        }
        for name in self.target_names:
            values = torch.full((self.num_objects,), float("nan"), dtype=torch.float32)
            values[:n_vertices] = self.vertex_targets[name][vs:ve]
            targets[f"particle_{name}"] = values
        targets["event_number"] = torch.tensor(self.event_number[idx], dtype=torch.int64)
        return inputs, targets


class SVDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        num_train: int,
        num_val: int,
        num_test: int,
        scale_dict_path: str,
        vertex_scale_dict_path: str,
        inputs: dict,
        targets: dict,
        train_path: str | None = None,
        valid_path: str | None = None,
        test_path: str | None = None,
        data_dir: str | None = None,
        split: list[int | float] | None = None,
        split_seed: int | None = 42,
        pin_memory: bool = True,
        test_suff: str | None = None,
        # SVDataset arguments, same names and defaults as SVDataset.__init__ (tests/test_sv_dataset.py checks the mirror)
        track_collection: str = TRACK_COLLECTION,
        vertex_collection: str = VERTEX_COLLECTION,
        num_objects: int = 16,
        max_tracks: int = 1024,
        min_tracks_for_valid: int = 1,
        track_min_pt: float = 0.0,
        track_max_abs_eta: float | None = None,
        track_require_pass_seed: bool = False,
        track_require_high_purity: bool = False,
        track_max_per_event: int | None = None,
        strict_scaling: bool = True,
        verbose: bool = False,
    ):
        """Each split reads either its own path (`train_path`/`valid_path`/`test_path`: a parquet file or a directory
        of them, e.g. the train/ val/ test/ folders of `maskformer-sv convert --split`) or its share of `data_dir`,
        cut into whole files by `split` after shuffling the file list with `split_seed` (maskformer.utils.parquet).
        An explicit path overrides the split for that split."""
        super().__init__()
        paths = {"train": train_path, "val": valid_path, "test": test_path}
        if data_dir is not None:
            if split is None:
                raise ValueError("data_dir needs a split, e.g. split: [0.8, 0.1, 0.1]")
            split_files = split_parquet_files(data_dir, split, seed=split_seed)
            print(f"Split the {sum(map(len, split_files.values()))} files used from {data_dir} into "
                  + ", ".join(f"{name} {len(files)}" for name, files in split_files.items()))
            for name, files in split_files.items():
                if paths[name] is not None:
                    print(f"{name}: explicit path {paths[name]} overrides the {len(files)} {name} files of data_dir")
                elif files:
                    paths[name] = [str(f) for f in files]
        elif split is not None:
            raise ValueError("split is only used together with data_dir")
        self.train_path, self.valid_path, self.test_path = paths["train"], paths["val"], paths["test"]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_train, self.num_val, self.num_test = num_train, num_val, num_test
        self.pin_memory = pin_memory
        self.test_suff = test_suff
        self.scale_dict_path = resolve_scale_dict_path(scale_dict_path)
        self.vertex_scale_dict_path = resolve_scale_dict_path(vertex_scale_dict_path)
        self.kwargs = dict(
            inputs=inputs,
            targets=targets,
            track_collection=track_collection,
            vertex_collection=vertex_collection,
            num_objects=num_objects,
            max_tracks=max_tracks,
            min_tracks_for_valid=min_tracks_for_valid,
            track_min_pt=track_min_pt,
            track_max_abs_eta=track_max_abs_eta,
            track_require_pass_seed=track_require_pass_seed,
            track_require_high_purity=track_require_high_purity,
            track_max_per_event=track_max_per_event,
            strict_scaling=strict_scaling,
            verbose=verbose,
        )

    def _dataset(self, filepath, num_events) -> SVDataset:
        return SVDataset(filepath=filepath, num_events=num_events, scale_dict_path=self.scale_dict_path,
                         vertex_scale_dict_path=self.vertex_scale_dict_path, **self.kwargs)

    def setup(self, stage: str):
        if stage == "fit":
            assert self.train_path is not None and self.valid_path is not None, \
                "No train/validation files, see --data.train_path/--data.valid_path or --data.data_dir/--data.split"
            self.train_dset = self._dataset(self.train_path, self.num_train)
            self.val_dset = self._dataset(self.valid_path, self.num_val)
            self.write_dataset_counts({"train": self.train_dset, "val": self.val_dset})
        if stage == "test":
            assert self.test_path is not None, "No test file specified, see --data.test_path or --data.data_dir/--data.split"
            self.test_dset = self._dataset(self.test_path, self.num_test)
            self.write_dataset_counts({"test": self.test_dset})

    def write_dataset_counts(self, datasets: dict):
        """Each dataset's counts and cutflow (SVDataset.counts) to <trainer.log_dir>/dataset_counts.json, keyed by split."""
        log_dir = self.trainer.log_dir  # broadcasts the directory, so call it on every rank
        counts = {split: dset.counts for split, dset in datasets.items() if hasattr(dset, "counts")}
        if not self.trainer.is_global_zero or not counts or log_dir is None:
            return
        path = Path(log_dir) / "dataset_counts.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            counts = {**json.loads(path.read_text()), **counts}
        path.write_text(json.dumps(counts, indent=2))
        print(f"Saved dataset counts and cutflow to {path}")

    def get_dataloader(self, dataset: SVDataset, shuffle: bool):
        return DataLoader(dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                          pin_memory=self.pin_memory)

    def train_dataloader(self):
        return self.get_dataloader(self.train_dset, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val_dset, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(self.test_dset, shuffle=False)


__all__ = ["NUM_CLASSES", "RESIDUAL_CLASS", "SVDataModule", "SVDataset"]
