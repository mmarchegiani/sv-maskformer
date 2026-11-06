#!/usr/bin/env python
"""Resolution of the regressed vertex properties: the residual of every variable the SV MaskFormer predicts, absolute
(pred - true) and relative ((pred - true) / true), under ``<outdir>/resolution/``.

For every regressed field of the prediction file (pt, eta, phi, sinphi, cosphi, lxy, lz; l3d, x, y, z when the model
regresses them; phi is the writer's atan2 of the predicted sin/cos, the truth of sinphi/cosphi the sin/cos of the
true phi) two histograms are written: ``residual_abs_<field>.png``, pred - true (wrapped into (-pi, pi] for phi),
and ``residual_rel_<field>.png``, (pred - true) / true (the vertices with true == 0 are dropped). ``overview.png``
puts them all on one page, one row per field. The window of a histogram is symmetric around zero, |median| +
``--range-sigmas`` sigma68 half-wide (sigma68 the robust width of the residual); the entries beyond it are collected
in the two edge bins and their fraction is written in the box with the median, mean, sigma68 and standard deviation.
The same numbers go to ``metrics["resolution"]`` (the chain's metrics.json and summary.txt; ``<outdir>/resolution/
metrics.json`` when the script runs on its own).

Which vertices enter (``--selection``): target vertices with at least EVAL_MIN_TRACKS selected tracks (the evaluation
selection of plot_sv_eval) that are ``loose``-matched (track recall and purity >= 0.5; the paper's and the
regression/ section's choice, the default), ``perfect``-matched (recall = purity = 1), ``found`` (the matched query
predicts a vertex, whatever its track mask) or all of them, ``targets`` (the regression is written for every target,
from the Hungarian-matched query). When the requested selection is empty, as for an untrained model, the next looser
one with at least one vertex is used and the plots and the metrics say so.

    python src/sv/scripts/plot/plot_sv_resolution.py --input <ckpt>__test.parquet --outdir <ckpt>__test_plots
    maskformer-sv plot --input <ckpt>__test.parquet --outdir <dir>        # as a section of the plot chain
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_sv_eval import EVAL_MIN_TRACKS  # noqa: E402
from sv_eval_common import CMS_blue, CMS_red, EvalData, robust_sigma, savefig, setup_style  # noqa: E402

plt = setup_style()

#: the regressed fields in the order of the plots: mathtext label (without the $) and unit
FIELDS = {
    "pt": (r"p_\mathrm{T}", "GeV"),
    "eta": (r"\eta", ""),
    "phi": (r"\phi", ""),
    "sinphi": (r"\sin\phi", ""),
    "cosphi": (r"\cos\phi", ""),
    "lxy": (r"L_{xy}", "cm"),
    "lz": (r"L_z", "cm"),
    "l3d": (r"L_{3D}", "cm"),
    "x": ("x", "cm"),
    "y": ("y", "cm"),
    "z": ("z", "cm"),
}
#: the two residuals of every field and their names in the summary
KINDS = ("abs", "rel")
KIND_LABELS = {"abs": "pred - true", "rel": "(pred - true) / true"}
#: the vertex selections, from the tightest to the loosest (the fallback walks down this tuple), and their names
SELECTIONS = ("perfect", "loose", "found", "targets")
SELECTION_LABELS = {
    "perfect": "perfect-matched target vertices (recall = purity = 1)",
    "loose": "loose-matched target vertices (recall, purity >= 0.5)",
    "found": "found target vertices (matched query predicts a vertex)",
    "targets": "all target vertices (Hungarian-matched query)",
}
DEFAULT_SELECTION = "loose"
DEFAULT_RANGE_SIGMAS = 5.0
N_BINS = 60


# --------------------------------------------------------------------------- #
# the numbers
# --------------------------------------------------------------------------- #


def select_vertices(d: EvalData, selection: str) -> tuple[np.ndarray, str]:
    """Mask of the vertices entering the plots and the selection actually used: the requested one, or the next looser
    one with at least one vertex."""
    if selection not in SELECTIONS:
        raise ValueError(f"unknown selection {selection!r}, expected one of {SELECTIONS}")
    base = d.v_is_target & (d.v_n_sel >= EVAL_MIN_TRACKS)
    masks = {"perfect": base & d.v_perfect, "loose": base & d.v_loose, "found": base & d.v_pred_valid, "targets": base}
    for used in SELECTIONS[SELECTIONS.index(selection):]:
        if masks[used].any():
            return masks[used], used
    return masks[selection], selection


def truth_and_prediction(d: EvalData, field: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Truth and prediction of one regressed field over all vertices (the truth of sinphi/cosphi from the true phi)."""
    pred = getattr(d, f"v_pred_{field}", None)
    if field in ("sinphi", "cosphi"):
        true = None if d.v_phi is None else (np.sin if field == "sinphi" else np.cos)(d.v_phi)
    else:
        true = getattr(d, f"v_{field}", None)
    return true, pred


def residual(true, pred, field: str, kind: str) -> np.ndarray:
    """``abs``: pred - true (wrapped into (-pi, pi] for phi); ``rel``: (pred - true) / true, NaN where true == 0."""
    true, pred = np.asarray(true, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    diff = pred - true
    if field == "phi":
        diff = np.arctan2(np.sin(diff), np.cos(diff))
    if kind == "abs":
        return diff
    if kind != "rel":
        raise ValueError(f"unknown residual kind {kind!r}, expected one of {KINDS}")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(true != 0, diff / true, np.nan)


def residual_window(res: np.ndarray, n_sigmas: float) -> tuple[float, float]:
    """Symmetric window around zero: half-width |median| + n_sigmas * sigma68 (at least 1.5 |median|, so that a
    narrow but biased residual stays inside); the 1-99 % quantiles when the robust width is not defined."""
    med, sig = float(np.median(res)), robust_sigma(res)
    w = max(abs(med) + n_sigmas * sig, 1.5 * abs(med)) if np.isfinite(sig) else float("nan")
    if not np.isfinite(w) or w <= 0:
        w = float(np.max(np.abs(np.percentile(res, [1, 99]))))
    if not np.isfinite(w) or w <= 0:
        w = 1.0
    return -w, w


def residual_stats(res: np.ndarray, window: tuple[float, float]) -> dict:
    lo, hi = window
    return {
        "n": int(len(res)),
        "median": float(np.median(res)),
        "mean": float(np.mean(res)),
        "sigma68": robust_sigma(res),
        "std": float(np.std(res)),
        "window": [float(lo), float(hi)],
        "beyond_window": float(np.mean((res < lo) | (res > hi))),
    }


# --------------------------------------------------------------------------- #
# the plots
# --------------------------------------------------------------------------- #


def axis_label(field: str, kind: str) -> str:
    lab, unit = FIELDS.get(field, (field, ""))
    if kind == "rel":
        return rf"$({lab}^{{\mathrm{{pred}}}} - {lab}^{{\mathrm{{true}}}})\,/\,{lab}^{{\mathrm{{true}}}}$"
    return rf"${lab}^{{\mathrm{{pred}}}} - {lab}^{{\mathrm{{true}}}}$" + (f" [{unit}]" if unit else "")


def draw_residual(ax, res: np.ndarray, field: str, kind: str, stats: dict, title: str, compact: bool = False) -> None:
    """One residual histogram on ``ax``; the entries beyond the window are collected in the two edge bins."""
    lo, hi = stats["window"]
    fs = 10 if compact else 14
    label_kw = {"fontsize": 13} if compact else {}
    ax.hist(np.clip(res, lo, hi), bins=np.linspace(lo, hi, N_BINS + 1), color=CMS_blue, alpha=0.8)
    ax.axvline(0.0, color="black", ls=":", lw=1.2, label="zero")
    ax.axvline(stats["median"], color=CMS_red, lw=1.5, label="median")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(axis_label(field, kind), **label_kw)
    ax.set_ylabel("vertices", **label_kw)
    if compact:  # the overview carries the selection in its suptitle
        ax.tick_params(labelsize=11)
    else:
        ax.set_title(title, loc="left", fontsize=fs)
    box = (f"n = {stats['n']}\nmedian {stats['median']:.3g}\nmean {stats['mean']:.3g}\n"
           f"$\\sigma_{{68}}$ {stats['sigma68']:.3g}\nstd {stats['std']:.3g}\nbeyond window {100 * stats['beyond_window']:.1f}%")
    ax.text(0.97, 0.95, box, transform=ax.transAxes, ha="right", va="top", fontsize=fs)
    ax.legend(loc="upper left", fontsize=fs)


def resolution_section(d: EvalData, outdir: str, metrics: dict, selection: str = DEFAULT_SELECTION,
                       range_sigmas: float = DEFAULT_RANGE_SIGMAS) -> dict:
    """The resolution/ section: the plots under ``<outdir>/resolution/`` and ``metrics["resolution"]`` (returned)."""
    out = os.path.join(outdir, "resolution")
    os.makedirs(out, exist_ok=True)
    fields = [f for f in FIELDS if f in d.regressed_fields]
    mask, used = select_vertices(d, selection)
    m = metrics["resolution"] = {
        "selection": used, "selection_requested": selection, "selection_label": SELECTION_LABELS[used],
        "n_vertices": int(mask.sum()), "range_sigmas": float(range_sigmas), "fields": fields,
    }
    if used != selection:
        print(f"[resolution] no {selection}-matched vertex with >= {EVAL_MIN_TRACKS} selected tracks: using the {used} selection", flush=True)
    if not mask.any() or not fields:
        print(f"[resolution] nothing to plot: {int(mask.sum())} vertices, regressed fields {fields}", flush=True)
        return m
    title = SELECTION_LABELS[used] + (f" [no {selection} match]" if used != selection else "")
    fig_all, axes = plt.subplots(len(fields), len(KINDS), figsize=(6.5 * len(KINDS), 4.2 * len(fields)), squeeze=False)
    for i, field in enumerate(fields):
        true, pred = truth_and_prediction(d, field)
        m[field] = {}
        for j, kind in enumerate(KINDS):
            res = np.zeros(0) if true is None or pred is None else residual(true[mask], pred[mask], field, kind)
            res = res[np.isfinite(res)]
            if not len(res):
                m[field][kind] = {"n": 0}
                axes[i, j].set_visible(False)
                continue
            stats = m[field][kind] = residual_stats(res, residual_window(res, range_sigmas))
            fig, ax = plt.subplots()
            draw_residual(ax, res, field, kind, stats, title)
            savefig(fig, out, f"residual_{kind}_{field}")
            draw_residual(axes[i, j], res, field, kind, stats, title, compact=True)
    fig_all.suptitle(f"{title}, n = {int(mask.sum())}", fontsize=14)
    fig_all.tight_layout()
    savefig(fig_all, out, "overview")
    return m


def summary_lines(m: dict) -> list[str]:
    """The lines of summary.txt for ``metrics["resolution"]``."""
    note = f"; no {m['selection_requested']}-matched vertex" if m.get("selection") != m.get("selection_requested") else ""
    lines = [f"Resolution ({m.get('selection_label', m.get('selection'))}, n = {m.get('n_vertices', 0)}{note}):"]
    rows = [(f, k, m[f][k]) for f in m.get("fields", ()) for k in KINDS if isinstance(m.get(f), dict) and m[f].get(k, {}).get("n")]
    if not rows:
        return lines + ["  no vertex to plot"]
    lines.append(f"  {'field':8s}{'residual':22s}{'n':>7s}{'median':>11s}{'mean':>11s}{'sigma68':>11s}{'std':>11s}{'beyond':>8s}")
    for f, k, s in rows:
        lines.append(f"  {f:8s}{KIND_LABELS[k]:22s}{s['n']:>7d}{s['median']:>+11.4g}{s['mean']:>+11.4g}{s['sigma68']:>11.4g}{s['std']:>11.4g}{100 * s['beyond_window']:>7.1f}%")
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="prediction parquet file (<ckpt>__test[_<suff>].parquet)")
    p.add_argument("--outdir", required=True, help="plots directory: the plots and metrics.json go to <outdir>/resolution/")
    p.add_argument("--selection", choices=SELECTIONS, default=DEFAULT_SELECTION, help="vertices entering the plots (default: %(default)s)")
    p.add_argument("--range-sigmas", type=float, default=DEFAULT_RANGE_SIGMAS,
                   help="half-width of the histogram window, in sigma68 of the residual beyond its median (default: %(default)s)")
    p.add_argument("--track-collection", default="track")
    p.add_argument("--vertex-collection", default="GV")
    p.add_argument("--max-events", type=int, default=None, help="use only the first N events")
    args = p.parse_args(argv)
    d = EvalData(args.input, max_events=args.max_events, track_collection=args.track_collection, vertex_collection=args.vertex_collection)
    print(f"[resolution] {d.summary()}", flush=True)
    m = resolution_section(d, args.outdir, {}, selection=args.selection, range_sigmas=args.range_sigmas)
    with open(os.path.join(args.outdir, "resolution", "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2, default=float)
    print("\n".join(summary_lines(m)))
    print(f"[resolution] done -> {os.path.join(args.outdir, 'resolution')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
