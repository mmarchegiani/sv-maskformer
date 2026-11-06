#!/usr/bin/env python
"""Evaluation of an SV MaskFormer prediction file, after section 4 of arXiv:2312.12272.

Produces, under ``--outdir``:

  efficiency/       vertex-finding efficiency, "perfect" (track recall = purity = 1) and "loose" (both >= 0.5), vs
                    the flight distance lxy, the hadron pT, the number of truth tracks, the number of vertices in the
                    event, and per truth category (b, c from b, prompt c, other); the recall / purity distributions
  fakes/            predicted vertices not matched to a truth vertex: per event, their track multiplicity, vs the
                    number of truth vertices
  classification/   the confusion matrix of the vertex classes (b, c, other) and the class probabilities
  regression/       residuals of every regressed vertex property, predicted vs true, resolution vs the true value
  tracks/           the track assignment: where the tracks of the heavy-flavour vertices end up, and how many
                    background tracks are claimed by a predicted vertex

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
    CMAP_10, CMS_blue, CMS_gray, CMS_orange, CMS_red, EvalData, binomial_error, category_color, eff_profile,
    robust_sigma, savefig, setup_style,
)

plt = setup_style()

#: minimum number of selected truth tracks for a vertex to enter the efficiencies (the paper's evaluation selection)
EVAL_MIN_TRACKS = 2
LXY_BINS = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
PT_BINS = np.array([10, 15, 20, 30, 40, 60, 80, 120, 200, 400])
NTRACK_BINS = np.arange(0.5, 9.5, 1.0)
NVERTEX_BINS = np.arange(-0.5, 9.5, 1.0)

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
    m["by_category"] = {}
    for c, cname in enumerate(d.category_names):
        s = sel & (d.v_category == c)
        if s.any():
            m["by_category"][cname] = {"n": int(s.sum()), "loose": float(d.v_loose[s].mean()), "perfect": float(d.v_perfect[s].mean())}

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

    # vs lxy per category
    fig, ax = plt.subplots()
    for c, cname in enumerate(d.category_names):
        s = sel & (d.v_category == c)
        if s.sum() >= 5:
            _eff_curve(ax, d.v_lxy[s], d.v_loose[s], LXY_BINS, f"{cname} (n={int(s.sum())})", category_color(c))
    ax.set_xscale("log")
    ax.set_xlabel(r"true $L_{xy}$ [cm]")
    ax.set_ylabel("loose vertex-finding efficiency")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    savefig(fig, out, "efficiency_vs_lxy_by_category")

    # per category bars
    fig, ax = plt.subplots()
    cats = [c for c in range(len(d.category_names)) if (sel & (d.v_category == c)).any()]
    xs = np.arange(len(cats))
    for off, (name, found, color) in enumerate((("loose", d.v_loose, CMS_blue), ("perfect", d.v_perfect, CMS_red))):
        vals = [found[sel & (d.v_category == c)].mean() for c in cats]
        errs = [binomial_error(found[sel & (d.v_category == c)].sum(), (sel & (d.v_category == c)).sum())[1] for c in cats]
        ax.bar(xs + (off - 0.5) * 0.35, vals, 0.35, yerr=errs, color=color, label=name, capsize=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d.category_names[c]}\n(n={int((sel & (d.v_category == c)).sum())})" for c in cats])
    ax.set_ylabel("vertex-finding efficiency")
    ax.set_ylim(0, 1.05)
    ax.legend()
    savefig(fig, out, "efficiency_by_category")

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


def write_summary(d: EvalData, metrics: dict, outdir: str) -> None:
    e, f, c, t = metrics["efficiency"], metrics["fakes"], metrics["classification"], metrics["tracks"]
    lines = [
        f"SV MaskFormer evaluation of {d.path}",
        f"  {d.summary()}",
        "",
        f"Vertex finding ({e['selection']}, n = {e['n_vertices']}):",
        f"  found (matched query predicts a vertex): {e['found']['efficiency']:.3f} +- {e['found']['error']:.3f}",
        f"  loose  (recall, purity >= 0.5):          {e['loose']['efficiency']:.3f} +- {e['loose']['error']:.3f}",
        f"  perfect (recall = purity = 1):           {e['perfect']['efficiency']:.3f} +- {e['perfect']['error']:.3f}",
    ]
    for cname, v in e.get("by_category", {}).items():
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
    efficiency_section(d, outdir, metrics)
    fakes_section(d, outdir, metrics)
    classification_section(d, outdir, metrics)
    regression_section(d, outdir, metrics)
    tracks_section(d, outdir, metrics)
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
