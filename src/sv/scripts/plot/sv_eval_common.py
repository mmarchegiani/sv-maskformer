"""Shared helpers of the SV evaluation plots: the loader of a prediction file and the plot style (torch-free).

The prediction files are the parquet dumps of `sv.callbacks.prediction_writer.SVPredictionWriter`: one record per
event, the schema of the test file plus the prediction branches on the vertex collection (`GV_pred_*`, one entry per
truth vertex, the output of the query the Hungarian matcher assigned to it), on the track collection
(`track_pred_match_idx`, `track_pred_slot`, `track_pred_prob`, `track_selected`) and the `fakeGV` collection (the
predicted vertices not matched to a truth vertex), see the writer's module doc.

`EvalData` flattens everything into numpy arrays over vertices, tracks and fakes (with the event of each entry), and
computes the per-vertex matching quantities of the paper (section 4): the track recall and purity of the matched
prediction, and the "perfect" (recall = purity = 1) and "loose" (both >= 0.5) matches.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq

_SRC_DIR = str(Path(__file__).resolve().parents[3])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from sv.dataset.collections import CLASS_NAMES, RESIDUAL_CLASS  # noqa: E402,F401

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #

CMAP_10 = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"]
CMS_blue, CMS_orange, CMS_red, CMS_gray, CMS_purple = CMAP_10[:5]
SAVE_PDF = False
_STYLE_ANNOUNCED = False


def setup_style():
    """Configure matplotlib (mplhep CMS style when available) and return ``pyplot``."""
    global _STYLE_ANNOUNCED
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import mplhep as hep
        hep.style.use("CMS")
        style = "mplhep CMS style"
    except ImportError:
        plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14,
                             "legend.fontsize": 13, "xtick.direction": "in", "ytick.direction": "in", "legend.frameon": False})
        style = "plain matplotlib style (mplhep not installed)"
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    if not _STYLE_ANNOUNCED:
        print(f"[style] {style}")
        _STYLE_ANNOUNCED = True
    return plt


def savefig(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def class_color(cls: int) -> str:
    return CMAP_10[cls % len(CMAP_10)]


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #


def binomial_error(k, n):
    """Binomial (Wald) uncertainty of k/n, 0 where n = 0."""
    k, n = np.asarray(k, dtype=np.float64), np.asarray(n, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, 0.0)
        err = np.where(n > 0, np.sqrt(p * (1 - p) / np.maximum(n, 1)), 0.0)
    return p, err


def eff_profile(x, passed, bins):
    """Efficiency of `passed` in bins of `x`: (centers, eff, err, counts)."""
    x = np.asarray(x, dtype=np.float64)
    passed = np.asarray(passed, dtype=bool)
    idx = np.digitize(x, bins) - 1
    n_bins = len(bins) - 1
    inside = (idx >= 0) & (idx < n_bins)
    n = np.bincount(idx[inside], minlength=n_bins)
    k = np.bincount(idx[inside], weights=passed[inside].astype(np.float64), minlength=n_bins)
    eff, err = binomial_error(k, n)
    centers = 0.5 * (bins[1:] + bins[:-1])
    return centers, eff, err, n


def robust_sigma(values):
    """Half the central 68% interval (the sigma of a Gaussian), robust against tails."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    lo, hi = np.percentile(values, [16, 84])
    return float(0.5 * (hi - lo))


# --------------------------------------------------------------------------- #
# the data
# --------------------------------------------------------------------------- #


def _flat(array) -> np.ndarray:
    return np.asarray(ak.to_numpy(ak.flatten(array)))


class EvalData:
    """One prediction file, flattened.

    Vertices (one entry per truth vertex of the file; `v_*` arrays, `v_event` the event of each):
      v_pt, v_eta, v_phi, v_lxy, v_lz, v_flight_xy (truth, from the file), v_class (B, BtoC, C, Other), v_n_tracks (tracks
      pointing to it in the file), v_n_sel (after the dataset's selection), v_is_target,
      v_pred_valid, v_pred_class, v_pred_prob (n, C+1), v_pred_n, v_pred_correct, v_recall, v_purity, v_perfect,
      v_loose, v_pred_<field> for every regressed field (un-scaled).
    Tracks (`t_*`): t_event, t_pt, t_eta, t_hadidx, t_label, t_selected, t_match (track_pred_match_idx), t_slot,
      t_prob.
    Fakes (`f_*`): f_event, f_class, f_prob, f_n_tracks, f_<field>.
    Events: n_events, e_n_vertices, e_n_targets, e_n_fakes.
    """

    def __init__(self, path, max_events=None, track_collection="track", vertex_collection="GV"):
        self.path = str(path)
        self.track_collection = track_collection
        self.vertex_collection = vertex_collection
        schema = pq.read_metadata(path).schema.to_arrow_schema().names
        columns = [c for c in (track_collection, vertex_collection, "fakeGV", "n_fake_vertices", "event_number") if c in schema]
        df = ak.from_arrow(pq.read_table(path, columns=columns))
        if max_events is not None:
            df = df[:max_events]
        self.n_events = len(df)
        gv, tr = df[vertex_collection], df[track_collection]
        V, T = f"{vertex_collection}_", f"{track_collection}_"
        self.regressed_fields = sorted({f[len(V) + 5:] for f in gv.fields if f.startswith(V + "pred_") and f[len(V) + 5:] in ("pt", "eta", "sinphi", "cosphi", "phi", "lxy", "lz", "l3d", "x", "y", "z")})
        self.class_names = list(CLASS_NAMES)

        # vertices
        n_v = np.asarray(ak.to_numpy(ak.num(gv[V + "pt"], axis=1)))
        self.e_n_vertices = n_v
        self.v_event = np.repeat(np.arange(self.n_events), n_v)
        for name in ("pt", "eta", "phi", "lxy", "lz", "l3d", "flight_xy", "x", "y", "z", "nDaughters"):
            setattr(self, f"v_{name}", _flat(gv[V + name]).astype(np.float64) if V + name in gv.fields else None)
        self.v_class = _flat(gv[V + "class"]).astype(np.int64)
        self.v_n_tracks = _flat(gv[V + "nTracks"]).astype(np.int64)
        self.v_n_sel = _flat(gv[V + "n_tracks_selected"]).astype(np.int64)
        self.v_is_target = _flat(gv[V + "is_target"]).astype(bool)
        self.v_pred_valid = _flat(gv[V + "pred_valid"]).astype(bool)
        self.v_pred_class = _flat(gv[V + "pred_class"]).astype(np.int64)
        prob_names = [*CLASS_NAMES, "null"]
        self.v_pred_prob = np.stack([_flat(gv[f"{V}pred_prob_{n}"]).astype(np.float64) for n in prob_names], axis=-1) if n_v.sum() else np.zeros((0, len(prob_names)))
        self.v_pred_n = _flat(gv[V + "pred_n_tracks"]).astype(np.int64)
        self.v_pred_correct = _flat(gv[V + "pred_n_correct"]).astype(np.int64)
        for field in self.regressed_fields:
            setattr(self, f"v_pred_{field}", _flat(gv[f"{V}pred_{field}"]).astype(np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            self.v_recall = np.where(self.v_n_sel > 0, self.v_pred_correct / np.maximum(self.v_n_sel, 1), 0.0)
            self.v_purity = np.where(self.v_pred_n > 0, self.v_pred_correct / np.maximum(self.v_pred_n, 1), 0.0)
        found = self.v_is_target & self.v_pred_valid
        self.v_perfect = found & (self.v_recall == 1) & (self.v_purity == 1)
        self.v_loose = found & (self.v_recall >= 0.5) & (self.v_purity >= 0.5)
        self.e_n_targets = np.bincount(self.v_event[self.v_is_target], minlength=self.n_events)

        # tracks
        n_t = np.asarray(ak.to_numpy(ak.num(tr[T + "pt"], axis=1)))
        self.t_event = np.repeat(np.arange(self.n_events), n_t)
        self.t_pt = _flat(tr[T + "pt"]).astype(np.float64)
        self.t_eta = _flat(tr[T + "eta"]).astype(np.float64)
        self.t_hadidx = _flat(tr[T + "hadidx"]).astype(np.int64)
        self.t_label = _flat(tr[T + "label"]).astype(np.int64) if T + "label" in tr.fields else None
        self.t_selected = _flat(tr[T + "selected"]).astype(bool)
        self.t_match = _flat(tr[T + "pred_match_idx"]).astype(np.int64)
        self.t_slot = _flat(tr[T + "pred_slot"]).astype(np.int64)
        self.t_prob = _flat(tr[T + "pred_prob"]).astype(np.float64)

        # fakes
        if "fakeGV" in df.fields and "fakeGV_slot" in df["fakeGV"].fields:
            fk = df["fakeGV"]
            n_f = np.asarray(ak.to_numpy(ak.num(fk["fakeGV_slot"], axis=1)))
            self.f_event = np.repeat(np.arange(self.n_events), n_f)
            self.f_class = _flat(fk["fakeGV_class"]).astype(np.int64)
            self.f_n_tracks = _flat(fk["fakeGV_n_tracks"]).astype(np.int64)
            self.f_prob = np.stack([_flat(fk[f"fakeGV_prob_{n}"]).astype(np.float64) for n in prob_names], axis=-1) if n_f.sum() else np.zeros((0, len(prob_names)))
            for field in self.regressed_fields:
                key = f"fakeGV_{field}"
                setattr(self, f"f_{field}", _flat(fk[key]).astype(np.float64) if key in fk.fields else None)
        else:
            n_f = np.zeros(self.n_events, dtype=np.int64)
            self.f_event = np.zeros(0, dtype=np.int64)
            self.f_class = np.zeros(0, dtype=np.int64)
            self.f_n_tracks = np.zeros(0, dtype=np.int64)
            self.f_prob = np.zeros((0, len(prob_names)))
        self.e_n_fakes = n_f.astype(np.int64)

    # ---- convenience ----------------------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return len(self.v_event)

    def summary(self) -> str:
        return (f"{self.n_events} events, {self.n_vertices} vertices ({int(self.v_is_target.sum())} targets), "
                f"{len(self.t_event)} tracks ({int(self.t_selected.sum())} selected), {int(self.e_n_fakes.sum())} fake vertices; "
                f"regressed: {self.regressed_fields}")
