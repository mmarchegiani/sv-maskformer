#!/usr/bin/env python
"""Evaluation of an SV MaskFormer prediction file, after section 4 of arXiv:2312.12272.

Produces, under ``--outdir``:

  composition/      class balance of the truth vertices (B, BtoC, C, Other) of the evaluated dataset, as pie charts: the
                    target vertices of the model, and those entering the efficiencies (>= EVAL_MIN_TRACKS tracks)
  efficiency/       vertex-finding efficiency, "perfect" (track recall = purity = 1) and "loose" (both >= 0.5), vs
                    the flight distance lxy, the hadron pT, the number of truth tracks, the number of vertices in the
                    event, and per truth class (B, BtoC, C, Other); the recall / purity distributions
  fakes/            predicted vertices not matched to a truth vertex: per event, their track multiplicity, vs the
                    number of truth vertices
  classification/   the confusion matrix of the vertex classes (B, BtoC, C, Other) and the class probabilities
  regression/       residuals of every regressed vertex property, predicted vs true, resolution vs the true value
  tracks/           the track assignment: where the tracks of the heavy-flavour vertices end up, and how many
                    background tracks are claimed by a predicted vertex
  roc/              fake rate vs efficiency for a scan of the working point (vertex-score cut x mask threshold x
                    minimum tracks per predicted vertex), with its Pareto front; needs the `slot` collection of the
                    prediction file (skipped for older files)
  dedup/            duplicate removal: non-maximum suppression of the predicted vertices on their shared tracks, then
                    a one-to-one loose re-matching to the targets (the training's Hungarian match often keeps the
                    duplicate and drops the matched copy); fake rate vs efficiency per overlap threshold

and ``metrics.json`` + ``summary.txt``; the chain (run_sv_plot_chain.py) adds resolution/, the absolute and relative
residual of every regressed property (plot_sv_resolution.py). A truth vertex enters the efficiencies when it was a
target of the model and has at least EVAL_MIN_TRACKS (2) selected tracks, as in the paper; the single-track vertices
are reported separately.

    python src/sv/scripts/plot/plot_sv_eval.py --input <ckpt>__test.parquet --outdir plots/sv_eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sv_eval_common import (  # noqa: E402
    CMAP_10, CMS_blue, CMS_gray, CMS_orange, CMS_red, EvalData, binomial_error, class_color, eff_profile,
    robust_sigma, savefig, setup_style,
)

plt = setup_style()

#: minimum number of selected truth tracks for a vertex to enter the efficiencies (the paper's evaluation selection)
EVAL_MIN_TRACKS = 2
LXY_BINS = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
PT_BINS = np.array([10, 15, 20, 30, 40, 60, 80, 120, 200, 400])
NTRACK_BINS = np.arange(0.5, 9.5, 1.0)
NVERTEX_BINS = np.arange(-0.5, 9.5, 1.0)

#: the working-point scan of roc_section: cut on the vertex score 1 - p(null), mask-probability threshold of the
#: track assignment and minimum number of tracks of a predicted vertex
ROC_SCORE_CUTS = np.r_[np.linspace(0.0, 0.95, 39), 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
ROC_MASK_THRESHOLDS = np.round(np.arange(0.1, 0.95, 0.1), 2)
ROC_MIN_TRACKS = (1, 2, 3)

#: duplicate removal of dedup_section: maximum shared fraction of the tracks of the smaller vertex (>= 1: none), the
#: reported default, the mask threshold of the track sets and the vertex-score cuts of the scan
DEDUP_OVERLAPS = (1.0, 0.8, 0.5, 0.3, 0.0)
DEDUP_DEFAULT_OVERLAP = 0.5
DEDUP_MASK_THRESHOLD = 0.5
DEDUP_SCORE_CUTS = np.round(np.r_[np.arange(0.0, 0.95, 0.05), 0.97, 0.99], 3)

#: regressed fields: label, unit, residual kind ("rel": (pred - true)/true, "abs": pred - true) and residual range
FIELDS = {
    "pt": dict(label=r"$p_\mathrm{T}$", unit="GeV", kind="rel", rng=(-1.0, 2.0)),
    "eta": dict(label=r"$\eta$", unit="", kind="abs", rng=(-0.5, 0.5)),
    "phi": dict(label=r"$\phi$", unit="", kind="abs", rng=(-0.5, 0.5)),
    "lxy": dict(label=r"$L_{xy}$", unit="cm", kind="rel", rng=(-1.0, 2.0)),
    "lz": dict(label=r"$L_z$", unit="cm", kind="abs", rng=(-1.0, 1.0)),
    "l3d": dict(label=r"$L_{3D}$", unit="cm", kind="rel", rng=(-1.0, 2.0)),
    "x": dict(label=r"$x$", unit="cm", kind="abs", rng=(-1.0, 1.0)),
    "y": dict(label=r"$y$", unit="cm", kind="abs", rng=(-1.0, 1.0)),
    "z": dict(label=r"$z$", unit="cm", kind="abs", rng=(-1.0, 1.0)),
}


def _eff_curve(ax, x, passed, bins, label, color, marker="o"):
    centers, eff, err, n = eff_profile(x, passed, bins)
    ok = n > 0
    ax.errorbar(centers[ok], eff[ok], yerr=err[ok], fmt=marker, color=color, label=label, capsize=2)
    return eff, n


def _wrap(phi):
    return np.arctan2(np.sin(phi), np.cos(phi))


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #


def composition_section(d: EvalData, outdir: str, metrics: dict) -> None:
    """Class balance of the truth vertices: one pie chart per selection, the counts in the metrics."""
    out = os.path.join(outdir, "composition")
    m = metrics["composition"] = {"classes": list(d.class_names)}
    for key, sel, title in (
        ("targets", d.v_is_target, "target vertices"),
        (f"targets_ge{EVAL_MIN_TRACKS}_tracks", d.v_is_target & (d.v_n_sel >= EVAL_MIN_TRACKS), f"target vertices with $\\geq$ {EVAL_MIN_TRACKS} tracks"),
    ):
        counts = np.bincount(d.v_class[sel], minlength=len(d.class_names))[:len(d.class_names)]
        total = int(counts.sum())
        m[key] = {"n_vertices": total, "counts": {c: int(n) for c, n in zip(d.class_names, counts)},
                  "fractions": {c: float(n / max(total, 1)) for c, n in zip(d.class_names, counts)}}
        if total == 0:
            continue
        shown = [c for c in range(len(d.class_names)) if counts[c] > 0]
        fig, ax = plt.subplots()
        wedges, _, _ = ax.pie(
            counts[shown], colors=[class_color(c) for c in shown], startangle=90, counterclock=False,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "", pctdistance=0.72,
            wedgeprops=dict(edgecolor="white", linewidth=2), textprops=dict(color="white", fontweight="bold"))
        ax.axis("equal")
        ax.legend(wedges, [f"{d.class_names[c]}: {int(counts[c])} ({100 * counts[c] / total:.1f}%)" for c in shown],
                  title=f"{total} {title}\n{d.n_events} events", loc="center left", bbox_to_anchor=(0.95, 0.5))
        savefig(fig, out, "class_balance" if key == "targets" else f"class_balance_ge{EVAL_MIN_TRACKS}_tracks")


def efficiency_section(d: EvalData, outdir: str, metrics: dict) -> None:
    out = os.path.join(outdir, "efficiency")
    sel = d.v_is_target & (d.v_n_sel >= EVAL_MIN_TRACKS)
    single = d.v_is_target & (d.v_n_sel == 1)
    m = metrics["efficiency"] = {"selection": f"target vertices with >= {EVAL_MIN_TRACKS} selected tracks", "n_vertices": int(sel.sum())}
    for name, found in (("loose", d.v_loose), ("perfect", d.v_perfect), ("found", d.v_is_target & d.v_pred_valid)):
        eff, err = binomial_error(found[sel].sum(), sel.sum())
        m[name] = {"efficiency": float(eff), "error": float(err)}
    if single.any():
        eff, err = binomial_error(d.v_loose[single].sum(), single.sum())
        m["single_track_vertices"] = {"n": int(single.sum()), "loose_efficiency": float(eff), "error": float(err)}
    m["by_class"] = {}
    for c, cname in enumerate(d.class_names):
        s = sel & (d.v_class == c)
        if s.any():
            m["by_class"][cname] = {"n": int(s.sum()), "loose": float(d.v_loose[s].mean()), "perfect": float(d.v_perfect[s].mean())}

    # vs lxy, pt, n tracks, n vertices
    for key, x, bins, xlabel, log in (
        ("lxy", d.v_lxy, LXY_BINS, r"true $L_{xy}$ [cm]", True),
        ("pt", d.v_pt, PT_BINS, r"true hadron $p_\mathrm{T}$ [GeV]", True),
        ("ntracks", d.v_n_sel, NTRACK_BINS, "truth tracks of the vertex (selected)", False),
        ("nvertices", d.e_n_targets[d.v_event], NVERTEX_BINS, "target vertices in the event", False),
    ):
        if x is None:
            continue
        fig, ax = plt.subplots()
        s = sel if key != "ntracks" else d.v_is_target
        _eff_curve(ax, x[s], d.v_loose[s], bins, "loose (recall, purity $\\geq$ 0.5)", CMS_blue)
        _eff_curve(ax, x[s], d.v_perfect[s], bins, "perfect (recall = purity = 1)", CMS_red, "s")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("vertex-finding efficiency")
        ax.set_ylim(0, 1.05)
        if log:
            ax.set_xscale("log")
        ax.legend(loc="lower right")
        savefig(fig, out, f"efficiency_vs_{key}")

    # vs lxy per class
    fig, ax = plt.subplots()
    for c, cname in enumerate(d.class_names):
        s = sel & (d.v_class == c)
        if s.sum() >= 5:
            _eff_curve(ax, d.v_lxy[s], d.v_loose[s], LXY_BINS, f"{cname} (n={int(s.sum())})", class_color(c))
    ax.set_xscale("log")
    ax.set_xlabel(r"true $L_{xy}$ [cm]")
    ax.set_ylabel("loose vertex-finding efficiency")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    savefig(fig, out, "efficiency_vs_lxy_by_class")

    # per class bars
    fig, ax = plt.subplots()
    cats = [c for c in range(len(d.class_names)) if (sel & (d.v_class == c)).any()]
    xs = np.arange(len(cats))
    for off, (name, found, color) in enumerate((("loose", d.v_loose, CMS_blue), ("perfect", d.v_perfect, CMS_red))):
        vals = [found[sel & (d.v_class == c)].mean() for c in cats]
        errs = [binomial_error(found[sel & (d.v_class == c)].sum(), (sel & (d.v_class == c)).sum())[1] for c in cats]
        ax.bar(xs + (off - 0.5) * 0.35, vals, 0.35, yerr=errs, color=color, label=name, capsize=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d.class_names[c]}\n(n={int((sel & (d.v_class == c)).sum())})" for c in cats])
    ax.set_ylabel("vertex-finding efficiency")
    ax.set_ylim(0, 1.05)
    ax.legend()
    savefig(fig, out, "efficiency_by_class")

    # recall / purity of the found vertices
    found = sel & d.v_pred_valid
    if found.any():
        fig, ax = plt.subplots()
        ax.hist(d.v_recall[found], bins=np.linspace(0, 1.0001, 21), histtype="step", color=CMS_blue, lw=2, label="track recall")
        ax.hist(d.v_purity[found], bins=np.linspace(0, 1.0001, 21), histtype="step", color=CMS_red, lw=2, label="track purity")
        ax.set_xlabel("per-vertex track recall / purity (found vertices)")
        ax.set_ylabel("vertices")
        ax.legend(loc="upper left")
        savefig(fig, out, "recall_purity")
        fig, ax = plt.subplots()
        h = ax.hist2d(d.v_recall[found], d.v_purity[found], bins=[np.linspace(0, 1.0001, 11)] * 2, cmap="Blues")
        fig.colorbar(h[3], ax=ax, label="vertices")
        ax.set_xlabel("track recall")
        ax.set_ylabel("track purity")
        savefig(fig, out, "recall_vs_purity")
        m["mean_recall_found"] = float(d.v_recall[found].mean())
        m["mean_purity_found"] = float(d.v_purity[found].mean())


def fakes_section(d: EvalData, outdir: str, metrics: dict) -> None:
    out = os.path.join(outdir, "fakes")
    m = metrics["fakes"] = {
        "n_fake_vertices": int(d.e_n_fakes.sum()),
        "per_event": float(d.e_n_fakes.mean()) if d.n_events else 0.0,
        "per_event_with_ge2_tracks": float((np.bincount(d.f_event[d.f_n_tracks >= 2], minlength=d.n_events)).mean()) if d.n_events else 0.0,
        "fraction_of_predicted_vertices": float(d.e_n_fakes.sum() / max(d.e_n_fakes.sum() + (d.v_is_target & d.v_pred_valid).sum(), 1)),
    }
    fig, ax = plt.subplots()
    ax.hist(d.e_n_fakes, bins=np.arange(-0.5, max(d.e_n_fakes.max(initial=0), 5) + 1.5), color=CMS_orange, alpha=0.8)
    ax.set_xlabel("fake vertices per event (predicted, unmatched)")
    ax.set_ylabel("events")
    ax.text(0.97, 0.95, f"mean {m['per_event']:.2f} / event", transform=ax.transAxes, ha="right", va="top")
    savefig(fig, out, "n_fakes_per_event")
    if len(d.f_n_tracks):
        fig, ax = plt.subplots()
        ax.hist(d.f_n_tracks, bins=np.arange(-0.5, max(d.f_n_tracks.max(), 5) + 1.5), color=CMS_orange, alpha=0.8)
        ax.set_xlabel("tracks of the fake vertex")
        ax.set_ylabel("fake vertices")
        savefig(fig, out, "fake_n_tracks")
        fig, ax = plt.subplots()
        for c, cname in enumerate(d.class_names):
            s = d.f_class == c
            if s.any():
                ax.hist(1 - d.f_prob[s, -1], bins=np.linspace(0, 1, 21), histtype="step", lw=2, color=CMAP_10[c], label=f"predicted {cname} (n={int(s.sum())})")
        ax.set_xlabel("1 - p(null) of the fake vertex")
        ax.set_ylabel("fake vertices")
        ax.set_yscale("log")
        ax.legend()
        savefig(fig, out, "fake_valid_prob")
        m["by_class"] = {cname: int((d.f_class == c).sum()) for c, cname in enumerate(d.class_names)}
    fig, ax = plt.subplots()
    bins = NVERTEX_BINS
    idx = np.digitize(d.e_n_targets, bins) - 1
    ok = (idx >= 0) & (idx < len(bins) - 1)
    n = np.bincount(idx[ok], minlength=len(bins) - 1)
    s = np.bincount(idx[ok], weights=d.e_n_fakes[ok], minlength=len(bins) - 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    ax.plot(centers[n > 0], s[n > 0] / n[n > 0], "o-", color=CMS_orange)
    ax.set_xlabel("target vertices in the event")
    ax.set_ylabel("fake vertices per event")
    savefig(fig, out, "fakes_vs_nvertices")


def classification_section(d: EvalData, outdir: str, metrics: dict) -> None:
    out = os.path.join(outdir, "classification")
    sel = d.v_is_target & (d.v_n_sel >= EVAL_MIN_TRACKS)
    n_cls = len(d.class_names)
    names = [*d.class_names, "null"]
    conf = np.zeros((n_cls, n_cls + 1), dtype=np.int64)
    for t in range(n_cls):
        for p in range(n_cls + 1):
            conf[t, p] = int((sel & (d.v_class == t) & (d.v_pred_class == p)).sum())
    m = metrics["classification"] = {"confusion_rows_true_cols_pred": conf.tolist(), "classes": names,
                                     "selection": "loose-matched target vertices"}
    matched = sel & d.v_loose
    if matched.any():
        m["accuracy_all"] = float((d.v_pred_class[matched] == d.v_class[matched]).mean())
        for t, cname in enumerate(d.class_names):
            s = matched & (d.v_class == t)
            if s.any():
                m[f"accuracy_{cname}"] = float((d.v_pred_class[s] == t).mean())
    fig, ax = plt.subplots()
    row_sum = conf.sum(1, keepdims=True)
    frac = np.where(row_sum > 0, conf / np.maximum(row_sum, 1), 0.0)
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for t in range(n_cls):
        for p in range(n_cls + 1):
            ax.text(p, t, f"{frac[t, p]:.2f}\n({conf[t, p]})", ha="center", va="center", color="black" if frac[t, p] < 0.6 else "white", fontsize=11)
    ax.set_xticks(range(n_cls + 1))
    ax.set_xticklabels(names)
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(d.class_names)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="fraction of the true class")
    savefig(fig, out, "confusion_matrix")
    for p, pname in enumerate(names):
        fig, ax = plt.subplots()
        for t, tname in enumerate(d.class_names):
            s = sel & (d.v_class == t)
            if s.any():
                ax.hist(d.v_pred_prob[s, p], bins=np.linspace(0, 1, 26), histtype="step", lw=2, color=CMAP_10[t], label=f"true {tname} (n={int(s.sum())})")
        ax.set_xlabel(f"p({pname})")
        ax.set_ylabel("vertices")
        ax.set_yscale("log")
        ax.legend()
        savefig(fig, out, f"prob_{pname}")


def regression_section(d: EvalData, outdir: str, metrics: dict) -> None:
    out = os.path.join(outdir, "regression")
    sel = d.v_is_target & (d.v_n_sel >= EVAL_MIN_TRACKS) & d.v_loose
    m = metrics["regression"] = {"selection": "loose-matched target vertices", "n_vertices": int(sel.sum())}
    if not sel.any():
        return
    for field in d.regressed_fields:
        if field in ("sinphi", "cosphi"):
            continue
        true = getattr(d, f"v_{field}", None)
        pred = getattr(d, f"v_pred_{field}", None)
        if true is None or pred is None or field not in FIELDS:
            continue
        spec = FIELDS[field]
        t, p = true[sel], pred[sel]
        if field == "phi":
            res = _wrap(p - t)
        elif spec["kind"] == "rel":
            with np.errstate(divide="ignore", invalid="ignore"):
                res = np.where(t != 0, (p - t) / t, np.nan)
        else:
            res = p - t
        res_ok = res[np.isfinite(res)]
        unit = f" [{spec['unit']}]" if spec["unit"] else ""
        rlabel = (f"({spec['label']}$^\\mathrm{{pred}}$ - {spec['label']}$^\\mathrm{{true}}$) / {spec['label']}$^\\mathrm{{true}}$"
                  if spec["kind"] == "rel" else f"{spec['label']}$^\\mathrm{{pred}}$ - {spec['label']}$^\\mathrm{{true}}${unit}")
        m[field] = {"median": float(np.median(res_ok)) if len(res_ok) else float("nan"),
                    "sigma68": robust_sigma(res_ok), "mean_abs": float(np.mean(np.abs(res_ok))) if len(res_ok) else float("nan"),
                    "within_50pct": float(np.mean(np.abs(res_ok) < 0.5)) if spec["kind"] == "rel" and len(res_ok) else None}
        fig, ax = plt.subplots()
        ax.hist(np.clip(res_ok, *spec["rng"]), bins=60, range=spec["rng"], color=CMS_blue, alpha=0.8)
        ax.set_xlabel(rlabel)
        ax.set_ylabel("vertices")
        ax.text(0.97, 0.95, f"median {m[field]['median']:.3f}\n$\\sigma_{{68}}$ {m[field]['sigma68']:.3f}", transform=ax.transAxes, ha="right", va="top")
        savefig(fig, out, f"residual_{field}")
        fig, ax = plt.subplots()
        log = field in ("pt", "lxy", "l3d")
        lo, hi = (max(np.percentile(t, 0.5), 1e-3), np.percentile(t, 99.5)) if log else (np.percentile(t, 0.5), np.percentile(t, 99.5))
        bins = np.geomspace(lo, hi, 40) if log else np.linspace(lo, hi, 40)
        h = ax.hist2d(np.clip(t, lo, hi), np.clip(p, lo, hi), bins=[bins, bins], cmap="Blues", cmin=1)
        ax.plot([lo, hi], [lo, hi], "--", color=CMS_gray)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        fig.colorbar(h[3], ax=ax, label="vertices")
        ax.set_xlabel(f"true {spec['label']}{unit}")
        ax.set_ylabel(f"predicted {spec['label']}{unit}")
        savefig(fig, out, f"pred_vs_true_{field}")
        # resolution vs the true value
        idx = np.digitize(t, bins) - 1
        centers, sig, med = [], [], []
        for b in range(len(bins) - 1):
            r = res[(idx == b) & np.isfinite(res)]
            if len(r) >= 10:
                centers.append(0.5 * (bins[b] + bins[b + 1]))
                sig.append(robust_sigma(r))
                med.append(np.median(r))
        if centers:
            fig, ax = plt.subplots()
            ax.plot(centers, sig, "o-", color=CMS_red, label=r"$\sigma_{68}$")
            ax.plot(centers, med, "s--", color=CMS_blue, label="median")
            if log:
                ax.set_xscale("log")
            ax.axhline(0, color=CMS_gray, lw=1)
            ax.set_xlabel(f"true {spec['label']}{unit}")
            ax.set_ylabel(rlabel)
            ax.legend()
            savefig(fig, out, f"resolution_vs_true_{field}")


def tracks_section(d: EvalData, outdir: str, metrics: dict) -> None:
    out = os.path.join(outdir, "tracks")
    sel = d.t_selected
    hf = sel & (d.t_hadidx >= 0)
    bkg = sel & (d.t_hadidx < 0)
    # only tracks of target vertices can be "correctly" assigned: look the target flag up per track
    v_off = np.concatenate([[0], np.cumsum(d.e_n_vertices)])
    hf_target = hf.copy()
    hf_target[hf] = d.v_is_target[v_off[d.t_event[hf]] + d.t_hadidx[hf]]
    correct = hf_target & (d.t_match == d.t_hadidx)
    other = hf_target & (d.t_match >= 0) & (d.t_match != d.t_hadidx)
    fake = hf_target & (d.t_match == -2)
    lost = hf_target & (d.t_match == -1)
    m = metrics["tracks"] = {
        "hf_tracks_of_targets": int(hf_target.sum()),
        "assigned_to_right_vertex": float(correct.sum() / max(hf_target.sum(), 1)),
        "assigned_to_other_vertex": float(other.sum() / max(hf_target.sum(), 1)),
        "assigned_to_fake": float(fake.sum() / max(hf_target.sum(), 1)),
        "unassigned": float(lost.sum() / max(hf_target.sum(), 1)),
        "background_tracks": int(bkg.sum()),
        "background_claimed_by_a_vertex": float((bkg & (d.t_match != -1)).sum() / max(bkg.sum(), 1)),
        "background_claimed_by_a_matched_vertex": float((bkg & (d.t_match >= 0)).sum() / max(bkg.sum(), 1)),
    }
    fig, ax = plt.subplots()
    labels = ["right vertex", "other vertex", "fake vertex", "unassigned"]
    vals = [m["assigned_to_right_vertex"], m["assigned_to_other_vertex"], m["assigned_to_fake"], m["unassigned"]]
    ax.bar(labels, vals, color=[CMS_blue, CMS_orange, CMS_red, CMS_gray])
    ax.set_ylabel("fraction of the heavy-flavour tracks")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    savefig(fig, out, "hf_track_assignment")
    if d.t_label is not None:
        fig, ax = plt.subplots()
        labs = sorted(set(d.t_label[hf_target].tolist()))
        ax.bar([str(l) for l in labs], [correct[hf_target & (d.t_label == l)].sum() / max((hf_target & (d.t_label == l)).sum(), 1) for l in labs], color=CMS_blue)
        ax.set_xlabel("track_label (2 b, 3 c from b, 4 prompt c)")
        ax.set_ylabel("fraction assigned to the right vertex")
        ax.set_ylim(0, 1.05)
        savefig(fig, out, "hf_track_assignment_by_label")
    fig, ax = plt.subplots()
    pt_bins = np.geomspace(0.8, 100, 20)
    _eff_curve(ax, d.t_pt[hf_target], correct[hf_target], pt_bins, "HF track to the right vertex", CMS_blue)
    _eff_curve(ax, d.t_pt[bkg], d.t_match[bkg] != -1, pt_bins, "background track claimed by a vertex", CMS_red, "s")
    ax.set_xscale("log")
    ax.set_xlabel(r"track $p_\mathrm{T}$ [GeV]")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.legend()
    savefig(fig, out, "track_assignment_vs_pt")
    fig, ax = plt.subplots()
    ax.hist(d.t_prob[hf_target & (d.t_match != -1)], bins=np.linspace(0, 1, 26), histtype="step", lw=2, color=CMS_blue, label="HF tracks (claimed)")
    ax.hist(d.t_prob[bkg & (d.t_match != -1)], bins=np.linspace(0, 1, 26), histtype="step", lw=2, color=CMS_red, label="background tracks (claimed)")
    ax.set_xlabel("mask probability of the claiming vertex")
    ax.set_ylabel("tracks")
    ax.set_yscale("log")
    ax.legend()
    savefig(fig, out, "track_prob")


# ---- fake rate vs efficiency ---------------------------------------------------------------


def _slot_truth(d: EvalData):
    """Per slot: the selected truth tracks of its matched target (0 if unmatched), whether the target enters the
    efficiency (>= EVAL_MIN_TRACKS tracks) and whether the slot is ignored (matched to a target below that)."""
    matched = d.s_vertex >= 0
    n_truth = np.zeros(len(matched), dtype=np.int64)
    n_truth[matched] = d.v_n_sel[d.s_vertex[matched]]
    evaluated = matched & (n_truth >= EVAL_MIN_TRACKS)
    return n_truth, evaluated, matched & ~evaluated


def _loose(n_pred, n_correct, n_truth, evaluated):
    """Loose match of a slot to its target: track recall and purity >= 0.5 (as EvalData.v_loose)."""
    return evaluated & (n_pred > 0) & (2 * n_correct >= n_truth) & (2 * n_correct >= n_pred)


def _rates(n_good, n_fake, n_targets, n_events) -> dict:
    """Efficiency and fake rate (with binomial errors) from the counts of loose-matched and other predicted vertices."""
    n_good, n_fake = np.asarray(n_good, dtype=np.float64), np.asarray(n_fake, dtype=np.float64)
    eff, eff_err = binomial_error(n_good, np.full_like(n_good, n_targets))
    fr, fr_err = binomial_error(n_fake, n_good + n_fake)
    empty = (n_good + n_fake) == 0  # no predicted vertex: the fake rate is undefined
    return {"n_matched": n_good, "n_fake": n_fake, "efficiency": eff, "efficiency_error": eff_err,
            "fake_rate": np.where(empty, np.nan, fr), "fake_rate_error": np.where(empty, np.nan, fr_err),
            "fakes_per_event": n_fake / max(n_events, 1)}


def scan_working_points(d: EvalData, score_cuts=ROC_SCORE_CUTS, mask_thresholds=ROC_MASK_THRESHOLDS, min_tracks=ROC_MIN_TRACKS) -> dict:
    """Efficiency and fake rate of every configuration (score cut t, mask threshold m, minimum tracks k), as flat arrays.

    A slot is a predicted vertex when 1 - p(null) > t and at least k tracks have a mask probability >= m. The
    efficiency is the fraction of the target vertices with >= EVAL_MIN_TRACKS selected tracks whose slot is a predicted
    vertex loosely matched to them (recall, purity >= 0.5 at the threshold m); the fake rate is the fraction of the
    predicted vertices that are not such a match (unmatched slots and matches of bad quality). The slots matched to a
    target below EVAL_MIN_TRACKS are left out of both. The Hungarian matching does not depend on (t, m, k).
    """
    score_cuts = np.asarray(score_cuts, dtype=np.float64)
    n_truth, evaluated, ignored = _slot_truth(d)
    n_slots = len(d.s_score)
    n_targets = int((d.v_is_target & (d.v_n_sel >= EVAL_MIN_TRACKS)).sum())
    cols = {"score_cut": [], "mask_threshold": [], "min_tracks": [], "n_good": [], "n_fake": []}
    for m in mask_thresholds:
        on = d.p_prob >= m
        n_pred = np.bincount(d.p_slot[on], minlength=n_slots)
        n_correct = np.bincount(d.p_slot[on & d.p_truth], minlength=n_slots)
        loose = _loose(n_pred, n_correct, n_truth, evaluated)
        for k in min_tracks:
            candidate = (n_pred >= k) & ~ignored
            for name, scores in (("n_good", d.s_score[candidate & loose]), ("n_fake", d.s_score[candidate & ~loose])):
                cols[name].append(len(scores) - np.searchsorted(np.sort(scores), score_cuts, side="right"))
            cols["score_cut"].append(score_cuts)
            cols["mask_threshold"].append(np.full(len(score_cuts), m))
            cols["min_tracks"].append(np.full(len(score_cuts), k))
    scan = {name: np.concatenate(values) if values else np.zeros(0) for name, values in cols.items()}
    n_good, n_fake = scan.pop("n_good"), scan.pop("n_fake")
    return {**scan, **_rates(n_good, n_fake, n_targets, d.n_events)}


def model_working_point(d: EvalData, min_tracks: int = 1) -> dict:
    """The same quantities at the working point of the model: the class argmax decides that a slot is a vertex, the
    tracks are those above the model's pred_threshold."""
    n_truth, evaluated, ignored = _slot_truth(d)
    loose = _loose(d.s_n_tracks, d.s_n_correct, n_truth, evaluated)
    predicted = d.s_pred_valid & (d.s_n_tracks >= min_tracks) & ~ignored
    n_targets = int((d.v_is_target & (d.v_n_sel >= EVAL_MIN_TRACKS)).sum())
    return {k: float(v) for k, v in _rates((predicted & loose).sum(), (predicted & ~loose).sum(), n_targets, d.n_events).items()}


def pareto_front(efficiency, fake_rate) -> np.ndarray:
    """Indices of the configurations no other one beats in both efficiency (higher) and fake rate (lower), by
    increasing efficiency. Configurations without a predicted vertex (NaN fake rate) are left out."""
    efficiency, fake_rate = np.asarray(efficiency, dtype=np.float64), np.asarray(fake_rate, dtype=np.float64)
    idx = np.flatnonzero(np.isfinite(fake_rate))
    order = idx[np.lexsort((fake_rate[idx], -efficiency[idx]))]  # efficiency descending, then fake rate ascending
    keep, best = [], np.inf
    for i in order:
        if fake_rate[i] < best:
            keep.append(i)
            best = fake_rate[i]
    return np.asarray(keep[::-1], dtype=np.int64)


def _roc_axes(ax, fake_rates):
    ax.set_xlabel(f"vertex-finding efficiency (loose, $\\geq$ {EVAL_MIN_TRACKS} tracks)")
    ax.set_ylabel("fake rate (predicted vertices, no loose match)")
    ax.set_xlim(0, 1.02)
    fake_rates = np.asarray(fake_rates, dtype=np.float64)
    if (fake_rates[np.isfinite(fake_rates)] > 0).any():
        ax.set_yscale("log")


def roc_section(d: EvalData, outdir: str, metrics: dict) -> None:
    out = os.path.join(outdir, "roc")
    if not getattr(d, "has_slots", False):
        metrics["roc"] = {"skipped": "the prediction file has no `slot` collection (written before the working-point scan): rerun the evaluation"}
        print("[roc] skipped: no `slot` collection in the prediction file")
        return
    scan = scan_working_points(d)
    eff, fr = scan["efficiency"], scan["fake_rate"]
    front = pareto_front(eff, fr)
    wps = {k: model_working_point(d, k) for k in ROC_MIN_TRACKS}
    config = lambda i: {name: float(values[i]) for name, values in scan.items()}  # noqa: E731
    metrics["roc"] = {
        "efficiency": f"target vertices with >= {EVAL_MIN_TRACKS} selected tracks whose slot is a predicted vertex loosely matched to them",
        "fake_rate": "predicted vertices without a loose match (unmatched slots and bad matches) / predicted vertices",
        "predicted_vertex": "1 - p(null) > score_cut and >= min_tracks tracks with mask probability >= mask_threshold",
        "n_configurations": int(len(eff)),
        "score_cuts": [float(x) for x in ROC_SCORE_CUTS], "mask_thresholds": [float(x) for x in ROC_MASK_THRESHOLDS], "min_tracks": list(ROC_MIN_TRACKS),
        "working_point": {f"min_tracks_{k}": wp for k, wp in wps.items()},
        "pareto_front": [config(i) for i in front],
    }
    os.makedirs(out, exist_ok=True)
    names = list(scan)
    np.savetxt(os.path.join(out, "scan.csv"), np.column_stack([scan[n] for n in names]), delimiter=",", header=",".join(names), comments="")

    def working_points(ax):
        for (k, wp), marker in zip(wps.items(), ("*", "P", "X")):
            ax.plot(wp["efficiency"], wp["fake_rate"], marker, color="black", ms=14, mfc="none", mew=1.5, ls="none",
                    label=f"model working point, $\\geq$ {k} track{'s' if k > 1 else ''}")

    # every configuration and the Pareto front
    fig, ax = plt.subplots()
    ok = np.isfinite(fr)
    ax.plot(eff[ok], fr[ok], ".", color=CMS_gray, ms=4, alpha=0.6, label=f"{int(ok.sum())} configurations")
    ax.plot(eff[front], fr[front], "-o", color=CMS_red, ms=4, lw=2, label="Pareto front")
    working_points(ax)
    _roc_axes(ax, fr)
    ax.legend(loc="upper left")
    savefig(fig, out, "fake_rate_vs_efficiency")

    # the score scan at the mask threshold closest to 0.5, per minimum number of tracks
    m_ref = float(ROC_MASK_THRESHOLDS[np.argmin(np.abs(ROC_MASK_THRESHOLDS - 0.5))])
    fig, ax = plt.subplots()
    for k, color in zip(ROC_MIN_TRACKS, (CMS_blue, CMS_orange, CMS_red)):
        s = (scan["mask_threshold"] == m_ref) & (scan["min_tracks"] == k) & ok
        ax.errorbar(eff[s], fr[s], xerr=scan["efficiency_error"][s], yerr=scan["fake_rate_error"][s], fmt="-o", ms=3, color=color,
                    label=f"score scan, $\\geq$ {k} track{'s' if k > 1 else ''}")
    working_points(ax)
    _roc_axes(ax, fr)
    ax.legend(loc="upper left", title=f"mask threshold {m_ref:g}")
    savefig(fig, out, "fake_rate_vs_efficiency_score_scan")

    # the score scan per mask threshold, at the evaluation's minimum number of tracks
    k_ref = EVAL_MIN_TRACKS if EVAL_MIN_TRACKS in ROC_MIN_TRACKS else ROC_MIN_TRACKS[0]
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")
    for j, m in enumerate(ROC_MASK_THRESHOLDS):
        s = (scan["mask_threshold"] == m) & (scan["min_tracks"] == k_ref) & ok
        ax.plot(eff[s], fr[s], "-", color=cmap(j / max(len(ROC_MASK_THRESHOLDS) - 1, 1)), lw=1.8, label=f"{m:g}")
    _roc_axes(ax, fr)
    ax.legend(loc="upper left", title=f"mask threshold ($\\geq$ {k_ref} tracks)", ncol=2)
    savefig(fig, out, "fake_rate_vs_efficiency_mask_thresholds")


def _slot_matrices(d: EvalData):
    """Per event: the range of its slots and the (slot, track) pair offsets, to rebuild the (Q, n_sel) matrices."""
    slot_off = np.concatenate([[0], np.cumsum(np.bincount(d.s_event, minlength=d.n_events))])
    pair_off = np.concatenate([[0], np.cumsum(np.bincount(d.p_slot, minlength=len(d.s_event)))])
    return slot_off, pair_off


def dedup_pass(d: EvalData, overlap_max: float, candidate: np.ndarray, mask_threshold: float = DEDUP_MASK_THRESHOLD,
               min_tracks: int = EVAL_MIN_TRACKS, slot_off=None, pair_off=None):
    """Duplicate removal and re-matching of the candidate slots, event by event.

    The predicted vertices are the candidate slots with >= min_tracks tracks of mask probability >= mask_threshold,
    by descending vertex score. Non-maximum suppression: a vertex is dropped when it shares more than overlap_max of
    the tracks of the smaller of the two with an already kept vertex (overlap_max >= 1: no suppression). The kept
    vertices are then matched one to one to the targets, greedily by descending score: a vertex matches the unmatched
    target with which it shares the most tracks, if the match is loose (recall, purity >= 0.5). Returns, per kept
    vertex: its score and its outcome (1 matched to an evaluated target, 0 fake, -1 matched to a target below
    EVAL_MIN_TRACKS, ignored as in the other sections). Both steps are greedy in score order, so the result for a
    score cut t is the subset of the vertices with score > t: one pass serves the whole score scan.
    """
    if slot_off is None:
        slot_off, pair_off = _slot_matrices(d)
    scores, outcome = [], []
    for e in range(d.n_events):
        a, b = slot_off[e], slot_off[e + 1]
        if b == a:
            continue
        n_pairs = pair_off[b] - pair_off[a]
        n = n_pairs // (b - a)
        on = (d.p_prob[pair_off[a]:pair_off[b]] >= mask_threshold).reshape(b - a, n)
        truth = d.p_truth[pair_off[a]:pair_off[b]].reshape(b - a, n)
        tgt = np.flatnonzero(d.s_vertex[a:b] >= 0)
        T, tn = truth[tgt], truth[tgt].sum(1)
        evaluated = d.v_n_sel[d.s_vertex[a:b][tgt]] >= EVAL_MIN_TRACKS
        n_on = on.sum(1)
        cand = np.flatnonzero(candidate[a:b] & (n_on >= min_tracks))
        cand = cand[np.argsort(-d.s_score[a:b][cand], kind="stable")]
        kept, used = [], np.zeros(len(tgt), dtype=bool)
        for q in cand:
            if overlap_max < 1.0 and any((on[q] & on[p]).sum() > overlap_max * min(n_on[q], n_on[p]) for p in kept):
                continue
            kept.append(q)
            n_cor = (T & on[q]).sum(1)
            ok = ~used & (tn > 0) & (2 * n_cor >= tn) & (2 * n_cor >= n_on[q])
            scores.append(d.s_score[a + q])
            if ok.any():
                j = np.flatnonzero(ok)[np.argmax(n_cor[ok])]
                used[j] = True
                outcome.append(1 if evaluated[j] else -1)
            else:
                outcome.append(0)
    return np.asarray(scores, dtype=np.float64), np.asarray(outcome, dtype=np.int64)


def dedup_section(d: EvalData, outdir: str, metrics: dict) -> None:
    out = os.path.join(outdir, "dedup")
    if not getattr(d, "has_slots", False):
        metrics["dedup"] = {"skipped": "the prediction file has no `slot` collection: rerun the evaluation"}
        print("[dedup] skipped: no `slot` collection in the prediction file")
        return
    slot_off, pair_off = _slot_matrices(d)
    n_targets = int((d.v_is_target & (d.v_n_sel >= EVAL_MIN_TRACKS)).sum())
    active = d.s_score >= 0
    cuts = np.asarray(DEDUP_SCORE_CUTS, dtype=np.float64)
    result = {"description": (
        f"predicted vertices with >= {EVAL_MIN_TRACKS} tracks of mask probability >= {DEDUP_MASK_THRESHOLD}; non-maximum suppression "
        "by vertex score (drop a vertex sharing > overlap_max of the tracks of the smaller one with a kept vertex); one-to-one "
        "greedy loose re-matching to the targets. fake rate = unmatched kept vertices / kept vertices"),
        "default_overlap_max": DEDUP_DEFAULT_OVERLAP, "working_point": {}, "scan": {}}
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")
    for j, o in enumerate(DEDUP_OVERLAPS):
        name = "none" if o >= 1 else f"{o:g}"
        # the model's working point: the class argmax decides that a slot is a vertex
        s, oc = dedup_pass(d, o, d.s_pred_valid & active, slot_off=slot_off, pair_off=pair_off)
        wp = {k: float(v) for k, v in _rates((oc == 1).sum(), (oc == 0).sum(), n_targets, d.n_events).items()}
        result["working_point"][name] = wp
        # the score scan: every active slot is a candidate, cut on 1 - p(null)
        s, oc = dedup_pass(d, o, active, slot_off=slot_off, pair_off=pair_off)
        order = np.argsort(s)
        good = np.r_[np.cumsum((oc[order] == 1)[::-1])[::-1], 0]
        fake = np.r_[np.cumsum((oc[order] == 0)[::-1])[::-1], 0]
        first = np.searchsorted(s[order], cuts, side="right")  # vertices with score > cut
        rates = _rates(good[first], fake[first], n_targets, d.n_events)
        result["scan"][name] = {"score_cut": cuts.tolist(), **{k: np.asarray(v).tolist() for k, v in rates.items()}}
        color = "black" if o >= 1 else cmap(j / max(len(DEDUP_OVERLAPS) - 1, 1))
        ax.plot(rates["efficiency"], rates["fake_rate"], "-o", ms=3, color=color, label=f"overlap > {name}" if o < 1 else "no duplicate removal")
        ax.plot(wp["efficiency"], wp["fake_rate"], "*", ms=14, mfc="none", mew=1.5, color=color)
    ax.set_xlabel("efficiency (loose, re-matched)")
    ax.set_ylabel("fake rate")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize="x-small", title=r"score scan ($\star$: model working point)")
    savefig(fig, out, "fake_rate_vs_efficiency")

    fig, ax = plt.subplots()
    for j, o in enumerate(DEDUP_OVERLAPS):
        name = "none" if o >= 1 else f"{o:g}"
        sc = result["scan"][name]
        color = "black" if o >= 1 else cmap(j / max(len(DEDUP_OVERLAPS) - 1, 1))
        ax.plot(sc["efficiency"], sc["fakes_per_event"], "-o", ms=3, color=color, label=f"overlap > {name}" if o < 1 else "no duplicate removal")
    ax.set_xlabel("efficiency (loose, re-matched)")
    ax.set_ylabel("fake vertices per event")
    ax.legend(loc="upper left", fontsize="x-small")
    savefig(fig, out, "fakes_per_event_vs_efficiency")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "scan.json"), "w") as fh:
        json.dump(result["scan"], fh, indent=1, default=float)
    metrics["dedup"] = {k: v for k, v in result.items() if k != "scan"}


def write_summary(d: EvalData, metrics: dict, outdir: str) -> None:
    e, f, c, t = metrics["efficiency"], metrics["fakes"], metrics["classification"], metrics["tracks"]
    comp = metrics.get("composition", {}).get("targets")
    lines = [
        f"SV MaskFormer evaluation of {d.path}",
        f"  {d.summary()}",
        "",
        *([f"Class balance of the {comp['n_vertices']} target vertices: "
           + ", ".join(f"{k} {comp['counts'][k]} ({100 * v:.1f}%)" for k, v in comp["fractions"].items()), ""] if comp else []),
        f"Vertex finding ({e['selection']}, n = {e['n_vertices']}):",
        f"  found (matched query predicts a vertex): {e['found']['efficiency']:.3f} +- {e['found']['error']:.3f}",
        f"  loose  (recall, purity >= 0.5):          {e['loose']['efficiency']:.3f} +- {e['loose']['error']:.3f}",
        f"  perfect (recall = purity = 1):           {e['perfect']['efficiency']:.3f} +- {e['perfect']['error']:.3f}",
    ]
    for cname, v in e.get("by_class", {}).items():
        lines.append(f"    {cname:10s} n={v['n']:6d}  loose {v['loose']:.3f}  perfect {v['perfect']:.3f}")
    if "single_track_vertices" in e:
        s = e["single_track_vertices"]
        lines.append(f"  single-track vertices (n = {s['n']}): loose {s['loose_efficiency']:.3f}")
    lines += [
        "",
        f"Fake vertices: {f['n_fake_vertices']} ({f['per_event']:.3f} per event, {f['per_event_with_ge2_tracks']:.3f} per event with >= 2 tracks), "
        f"{f['fraction_of_predicted_vertices']:.3f} of the predicted vertices",
        "",
        "Classification accuracy (loose-matched vertices): " + (", ".join(f"{k.replace('accuracy_', '')} {v:.3f}" for k, v in c.items() if k.startswith("accuracy")) or "no loose-matched vertex"),
        "",
        "Regression (loose-matched vertices):",
    ]
    for field, v in metrics["regression"].items():
        if isinstance(v, dict):
            lines.append(f"  {field:6s} median {v['median']:+.4f}  sigma68 {v['sigma68']:.4f}  <|res|> {v['mean_abs']:.4f}"
                         + (f"  within 50%: {v['within_50pct']:.3f}" if v.get("within_50pct") is not None else ""))
    if metrics.get("roc", {}).get("working_point"):
        lines += ["", "Working point (fake rate = predicted vertices without a loose match / predicted vertices; scan in roc/):"]
        for name, wp in metrics["roc"]["working_point"].items():
            lines.append(f"  {name.replace('min_tracks_', '>= ')} track(s): efficiency {wp['efficiency']:.3f}, fake rate {wp['fake_rate']:.3f}, "
                         f"{wp['fakes_per_event']:.3f} fakes per event")
    if metrics.get("dedup", {}).get("working_point"):
        dd = metrics["dedup"]
        lines += ["", f"Duplicate removal + re-matching (model working point, >= {EVAL_MIN_TRACKS} tracks; scan in dedup/):"]
        for name, wp in dd["working_point"].items():
            label = "no removal  " if name == "none" else f"overlap > {name:4s}"
            lines.append(f"  {label}: efficiency {wp['efficiency']:.3f}, fake rate {wp['fake_rate']:.3f}, {wp['fakes_per_event']:.3f} fakes per event"
                         + ("   <- default" if name == f"{dd['default_overlap_max']:g}" else ""))
    if metrics.get("resolution"):
        from plot_sv_resolution import summary_lines  # lazily: that module imports EVAL_MIN_TRACKS from this one

        lines += ["", *summary_lines(metrics["resolution"])]
    lines += [
        "",
        f"Tracks: {t['hf_tracks_of_targets']} heavy-flavour tracks of target vertices: right vertex {t['assigned_to_right_vertex']:.3f}, "
        f"other vertex {t['assigned_to_other_vertex']:.3f}, fake {t['assigned_to_fake']:.3f}, unassigned {t['unassigned']:.3f}",
        f"        {t['background_tracks']} background tracks: claimed by a vertex {t['background_claimed_by_a_vertex']:.4f} "
        f"(by a matched one {t['background_claimed_by_a_matched_vertex']:.4f})",
    ]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        fh.write(text)
    print(text)


def run(d: EvalData, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    metrics: dict = {"input": d.path, "n_events": d.n_events, "n_vertices": d.n_vertices, "eval_min_tracks": EVAL_MIN_TRACKS}
    composition_section(d, outdir, metrics)
    efficiency_section(d, outdir, metrics)
    fakes_section(d, outdir, metrics)
    classification_section(d, outdir, metrics)
    regression_section(d, outdir, metrics)
    tracks_section(d, outdir, metrics)
    roc_section(d, outdir, metrics)
    dedup_section(d, outdir, metrics)
    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    write_summary(d, metrics, outdir)
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--track-collection", default="track")
    p.add_argument("--vertex-collection", default="GV")
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args(argv)
    d = EvalData(args.input, max_events=args.max_events, track_collection=args.track_collection, vertex_collection=args.vertex_collection)
    run(d, args.outdir)


if __name__ == "__main__":
    main()
