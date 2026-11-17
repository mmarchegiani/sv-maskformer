#!/usr/bin/env python
"""Study of the dynamic queries of an SV MaskFormer (configs/sv/gv_tracks_dynamic_queries.yaml): how good the
`query_init` seed task is, and how the number of queries relates to the number of true vertices and to the fakes.

Needs a prediction file of a dynamic-query model written with `track_seed_prob` / `track_seed` and a correct
`slot_active` (SVPredictionWriter after the query_mask fix); the query-count part (2) also runs on a fixed-query
file. Produces, under ``--outdir``:

  seeds/      (1) the seed task: seed-probability distributions of the seed tracks, the other tracks of the target
              vertices and the background tracks; vs the seed threshold (the queries are the tracks with
              p >= threshold, at most --num-queries, the most probable): queries per event, seed recall (the
              target's seed track is a query), vertex coverage (at least one track of the target is a query),
              duplicate queries (a second query on a covered target) and background queries per event
  queries/    (2) the number of active queries, of predicted vertices and of fakes vs the number of target vertices,
              the fakes vs the query excess (queries - targets), the vertex score 1 - p(null) of the matched and of
              the unmatched active slots, and what the fakes are made of (tracks of target vertices = split /
              duplicate vertices, or background tracks)

and ``queries_metrics.json`` + ``queries_summary.txt``.

    python src/sv/scripts/plot/plot_sv_queries.py --input <ckpt>__test.parquet --outdir plots/sv_queries \\
        --num-queries 12 --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sv_eval_common import CMS_blue, CMS_gray, CMS_orange, CMS_red, binomial_error, savefig, setup_style  # noqa: E402

plt = setup_style()

THRESHOLDS = np.round(np.r_[0.05, np.arange(0.1, 0.95, 0.05), 0.95, 0.97, 0.99], 3)
COUNT_BINS = np.arange(-0.5, 16.5, 1.0)


def load(path, track_collection="track", max_events=None):
    import awkward as ak
    import pyarrow.parquet as pq

    columns = [track_collection, "GV", "slot"]
    table = pq.read_table(path, columns=[c for c in columns if c in pq.read_schema(path).names])
    a = ak.from_arrow(table)
    if max_events is not None:
        a = a[:max_events]
    T = track_collection + "_"
    tr, sl = a[track_collection], a["slot"]
    has_seeds = T + "seed_prob" in tr.fields
    ev = []
    for i in range(len(a)):
        sel = np.asarray(tr[T + "selected"][i], dtype=bool)
        e = {
            "active": np.asarray(sl["slot_active"][i], dtype=bool),
            "matched": np.asarray(sl["slot_target_idx"][i]) >= 0,
            "pred_valid": np.asarray(sl["slot_pred_valid"][i], dtype=bool),
            "score": 1.0 - np.asarray(sl["slot_prob_null"][i], dtype=np.float64),
            "mask_prob": np.asarray(ak.to_numpy(sl["slot_mask_prob"][i])) if len(sl["slot_mask_prob"][i]) else np.zeros((0, sel.sum())),
            "truth": np.asarray(ak.to_numpy(sl["slot_truth_mask"][i])) if len(sl["slot_truth_mask"][i]) else np.zeros((0, sel.sum()), dtype=bool),
            "n_targets": int(np.sum(np.asarray(a["GV"]["GV_is_target"][i], dtype=bool))),
        }
        # the target vertex of every selected track (the slot of its matched query; -1 = background). The mask fields
        # are in the order of the selected tracks, as track_* restricted to track_selected (data.sort_tracks_by null)
        truth = e["truth"]
        e["owner"] = np.where(truth.any(0), truth.argmax(0), -1) if truth.size else np.full(int(sel.sum()), -1)
        if has_seeds:
            e["seed_prob"] = np.asarray(tr[T + "seed_prob"][i], dtype=np.float64)[sel]
            e["seed"] = np.asarray(tr[T + "seed"][i], dtype=bool)[sel]
        ev.append(e)
    return ev, has_seeds


def choose_queries(p, threshold, k):
    """hepattn's MaskFormerDecoder.initialize_dynamic_queries: the tracks with p >= threshold, at most k (the most
    probable); the top k when none passes."""
    idx = np.flatnonzero(p >= threshold)
    if idx.size == 0:
        idx = np.argsort(-p)[:k]
    elif idx.size > k:
        idx = idx[np.argsort(-p[idx])[:k]]
    return idx


# ------------------------------------------------------------------------------------------------- (1) seeds


def seed_scan(ev, k, thresholds=THRESHOLDS):
    rows = []
    for t in thresholds:
        nq = ntgt = seed_hit = covered = dup = bkg = 0
        for e in ev:
            q = choose_queries(e["seed_prob"], t, k)
            owner = e["owner"]
            targets = np.unique(owner[owner >= 0])
            seed_owner = owner[e["seed"]]
            q_owner = owner[q]
            n_cov = len(np.unique(q_owner[q_owner >= 0]))
            nq += len(q)
            ntgt += len(targets)
            seed_hit += int(np.isin(np.flatnonzero(e["seed"]), q).sum()) if len(seed_owner) else 0
            covered += n_cov
            dup += int((q_owner >= 0).sum()) - n_cov
            bkg += int((q_owner < 0).sum())
        n = max(len(ev), 1)
        rows.append({"threshold": float(t), "queries_per_event": nq / n, "targets_per_event": ntgt / n,
                     "seed_recall": seed_hit / max(ntgt, 1), "coverage": covered / max(ntgt, 1),
                     "duplicates_per_event": dup / n, "background_per_event": bkg / n,
                     "missed_targets_per_event": (ntgt - covered) / n})
    return rows


def seeds_section(ev, k, threshold, outdir, metrics):
    out = os.path.join(outdir, "seeds")
    p = np.concatenate([e["seed_prob"] for e in ev])
    seed = np.concatenate([e["seed"] for e in ev])
    hf = np.concatenate([e["owner"] >= 0 for e in ev])
    groups = {"seed tracks": (seed, CMS_red), "other tracks of target vertices": (hf & ~seed, CMS_orange),
              "background tracks": (~hf, CMS_gray)}
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 51)
    for label, (sel, color) in groups.items():
        ax.hist(p[sel], bins=bins, histtype="step", lw=2, color=color, density=True, label=f"{label} ({sel.sum()})")
    ax.axvline(threshold, color="black", ls="--", label=f"threshold {threshold}")
    ax.set_yscale("log")
    ax.set_xlabel("query_init seed probability")
    ax.set_ylabel("normalised")
    ax.legend(fontsize="x-small")
    savefig(fig, out, "seed_prob")

    # ranking quality: where the seed of each target ranks among the event's tracks
    ranks = []
    for e in ev:
        order = np.argsort(-e["seed_prob"])
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        ranks.append(rank[e["seed"]])
    ranks = np.concatenate(ranks) if ranks else np.zeros(0, dtype=int)
    fig, ax = plt.subplots()
    ax.hist(np.clip(ranks, 0, 40), bins=np.arange(-0.5, 41.5, 1), histtype="stepfilled", color=CMS_blue, alpha=0.7)
    ax.axvline(k - 0.5, color="black", ls="--", label=f"num_queries = {k}")
    ax.set_xlabel("rank of the seed track in its event (by seed probability, 40 = overflow)")
    ax.set_ylabel("seed tracks")
    ax.legend()
    savefig(fig, out, "seed_rank")

    scan = seed_scan(ev, k)
    x = [r["threshold"] for r in scan]
    fig, ax = plt.subplots()
    for key, label, color in (("queries_per_event", "queries", CMS_blue), ("targets_per_event", "target vertices", "black"),
                              ("duplicates_per_event", "duplicate queries (target already covered)", CMS_orange),
                              ("background_per_event", "background queries", CMS_gray),
                              ("missed_targets_per_event", "targets without a query", CMS_red)):
        ax.plot(x, [r[key] for r in scan], "o-", ms=4, color=color, label=label)
    ax.axvline(threshold, color="black", ls=":")
    ax.set_xlabel("seed threshold")
    ax.set_ylabel("per event")
    ax.legend(fontsize="x-small")
    savefig(fig, out, "per_event_vs_threshold")

    fig, ax = plt.subplots()
    ax.plot(x, [r["seed_recall"] for r in scan], "o-", ms=4, color=CMS_red, label="seed recall (seed track is a query)")
    ax.plot(x, [r["coverage"] for r in scan], "o-", ms=4, color=CMS_blue, label="coverage (a track of the target is a query)")
    ax.axvline(threshold, color="black", ls=":")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("seed threshold")
    ax.set_ylabel("fraction of target vertices")
    ax.legend(fontsize="x-small")
    savefig(fig, out, "recall_vs_threshold")

    # consistency: the recomputed query count at the model's threshold must equal the active slots
    nq = np.array([len(choose_queries(e["seed_prob"], threshold, k)) for e in ev])
    na = np.array([e["active"].sum() for e in ev])
    metrics["seeds"] = {"scan": scan, "n_tracks": int(len(p)), "n_seeds": int(seed.sum()),
                        "seed_rank_lt_num_queries": float(np.mean(ranks < k)) if len(ranks) else float("nan"),
                        "query_count_matches_slot_active": float(np.mean(nq == na))}


# ------------------------------------------------------------------------------------------------ (2) queries


def hist2d(ax, x, y, xlabel, ylabel):
    h, _, _ = np.histogram2d(x, y, bins=[COUNT_BINS, COUNT_BINS])
    col = h.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        hn = np.where(col > 0, h / col, np.nan)
    im = ax.pcolormesh(COUNT_BINS, COUNT_BINS, hn.T, cmap="viridis", vmin=0, vmax=1)
    xs = np.arange(len(COUNT_BINS) - 1)
    mean = np.array([y[x == v].mean() if np.any(x == v) else np.nan for v in xs])
    ax.plot(xs, mean, "o", color="white", mec="black", label="mean")
    ax.plot(xs, xs, ls="--", color="white", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize="x-small")
    plt.colorbar(im, ax=ax, label="fraction of events in the column")
    ax.set_xlim(COUNT_BINS[0], 8.5)


def queries_section(ev, outdir, metrics):
    out = os.path.join(outdir, "queries")
    n_tgt = np.array([e["n_targets"] for e in ev])
    n_act = np.array([e["active"].sum() for e in ev])
    n_pred = np.array([(e["pred_valid"] & e["active"]).sum() for e in ev])
    fake = [e["pred_valid"] & e["active"] & ~e["matched"] for e in ev]
    n_fake = np.array([f.sum() for f in fake])
    n_unm = np.array([(e["active"] & ~e["matched"]).sum() for e in ev])

    for y, name, label in ((n_act, "active_vs_targets", "active queries"), (n_pred, "predicted_vs_targets", "predicted vertices"),
                           (n_fake, "fakes_vs_targets", "fake vertices")):
        fig, ax = plt.subplots()
        hist2d(ax, n_tgt, y, "target vertices", label)
        savefig(fig, out, name)

    excess = n_act - n_tgt
    fig, ax = plt.subplots()
    xs = np.arange(excess.min(), excess.max() + 1) if len(excess) else np.zeros(0)
    ax.plot(xs, [n_fake[excess == v].mean() if np.any(excess == v) else np.nan for v in xs], "o-", color=CMS_red, label="fakes")
    ax.plot(xs, xs, "--", color="black", label="fakes = excess (no unmatched query rejected)")
    ax.set_xlabel("query excess (active queries - target vertices)")
    ax.set_ylabel("mean fake vertices per event")
    ax.legend(fontsize="x-small")
    savefig(fig, out, "fakes_vs_excess")

    # does the null class separate the unmatched active queries from the matched ones?
    s_m = np.concatenate([e["score"][e["active"] & e["matched"]] for e in ev])
    s_u = np.concatenate([e["score"][e["active"] & ~e["matched"]] for e in ev])
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 41)
    ax.hist(s_m, bins=bins, histtype="step", lw=2, density=True, color=CMS_blue, label=f"matched to a target ({len(s_m)})")
    ax.hist(s_u, bins=bins, histtype="step", lw=2, density=True, color=CMS_red, label=f"unmatched ({len(s_u)})")
    ax.set_xlabel("vertex score 1 - p(null), active slots")
    ax.set_ylabel("normalised")
    ax.legend(fontsize="x-small")
    savefig(fig, out, "score_matched_vs_unmatched")

    # composition of the fakes: the fraction of their tracks (mask prob >= 0.5) that belong to target vertices
    frac, n_tr, n_owners = [], [], []
    for e, f in zip(ev, fake):
        for q in np.flatnonzero(f):
            on = e["mask_prob"][q] >= 0.5
            if not on.any():
                continue
            own = e["owner"][on]
            frac.append(np.mean(own >= 0))
            n_tr.append(int(on.sum()))
            n_owners.append(len(np.unique(own[own >= 0])))
    frac, n_owners = np.asarray(frac), np.asarray(n_owners)
    fig, ax = plt.subplots()
    ax.hist(frac, bins=np.linspace(0, 1, 11), histtype="stepfilled", color=CMS_orange, alpha=0.8)
    ax.set_xlabel("fraction of the fake's tracks that belong to target vertices")
    ax.set_ylabel("fake vertices")
    savefig(fig, out, "fake_composition")

    rej, rej_err = binomial_error(n_unm.sum() - n_fake.sum(), n_unm.sum())
    metrics["queries"] = {
        "events": int(len(ev)), "targets_per_event": float(n_tgt.mean()), "active_queries_per_event": float(n_act.mean()),
        "predicted_per_event": float(n_pred.mean()), "fakes_per_event": float(n_fake.mean()),
        "unmatched_active_per_event": float(n_unm.mean()),
        "null_rejection_of_unmatched_queries": float(rej), "null_rejection_error": float(rej_err),
        "events_fewer_queries_than_targets": float(np.mean(n_act < n_tgt)),
        "fakes_mostly_target_tracks": float(np.mean(frac >= 0.5)) if len(frac) else float("nan"),
        "fakes_only_background_tracks": float(np.mean(frac == 0)) if len(frac) else float("nan"),
        "fakes_spanning_two_or_more_targets": float(np.mean(n_owners >= 2)) if len(n_owners) else float("nan"),
        "fakes_per_event_by_targets": {int(v): float(n_fake[n_tgt == v].mean()) for v in np.unique(n_tgt)},
        "active_per_event_by_targets": {int(v): float(n_act[n_tgt == v].mean()) for v in np.unique(n_tgt)},
        "matched_score_median": float(np.median(s_m)) if len(s_m) else float("nan"),
        "unmatched_score_median": float(np.median(s_u)) if len(s_u) else float("nan"),
    }


def summary(metrics) -> str:
    lines = []
    q = metrics["queries"]
    lines.append(f"Queries ({q['events']} events): {q['targets_per_event']:.2f} targets, {q['active_queries_per_event']:.2f} active queries, "
                 f"{q['predicted_per_event']:.2f} predicted vertices, {q['fakes_per_event']:.2f} fakes per event")
    lines.append(f"  unmatched active queries per event {q['unmatched_active_per_event']:.2f}; rejected by the null class: "
                 f"{q['null_rejection_of_unmatched_queries']:.3f} +- {q['null_rejection_error']:.3f}")
    lines.append(f"  events with fewer queries than targets: {q['events_fewer_queries_than_targets']:.3f}")
    lines.append(f"  fakes: mostly target-vertex tracks {q['fakes_mostly_target_tracks']:.3f}, background only "
                 f"{q['fakes_only_background_tracks']:.3f}, spanning >= 2 targets {q['fakes_spanning_two_or_more_targets']:.3f}")
    lines.append(f"  vertex score median: matched {q['matched_score_median']:.3f}, unmatched {q['unmatched_score_median']:.3f}")
    lines.append("  targets: " + ", ".join(f"{k}: {q['active_per_event_by_targets'][k]:.1f} queries / {v:.2f} fakes"
                                          for k, v in q["fakes_per_event_by_targets"].items()))
    if "seeds" in metrics:
        s = metrics["seeds"]
        lines.append(f"Seeds: {s['n_seeds']} seed tracks of {s['n_tracks']}; seed ranked below num_queries: "
                     f"{s['seed_rank_lt_num_queries']:.3f}; recomputed query count == slot_active: {s['query_count_matches_slot_active']:.3f}")
        lines.append("  threshold  queries  targets  seed_recall  coverage  duplicates  background  missed")
        for r in s["scan"]:
            lines.append(f"  {r['threshold']:9.2f}  {r['queries_per_event']:7.2f}  {r['targets_per_event']:7.2f}  {r['seed_recall']:11.3f}  "
                         f"{r['coverage']:8.3f}  {r['duplicates_per_event']:10.2f}  {r['background_per_event']:10.2f}  {r['missed_targets_per_event']:6.2f}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--num-queries", type=int, default=12, help="decoder.num_queries of the model")
    p.add_argument("--threshold", type=float, default=0.5, help="threshold of the query_init task of the model")
    p.add_argument("--track-collection", default="track")
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args(argv)

    ev, has_seeds = load(args.input, args.track_collection, args.max_events)
    metrics = {"input": args.input, "num_queries": args.num_queries, "threshold": args.threshold}
    queries_section(ev, args.outdir, metrics)
    if has_seeds:
        seeds_section(ev, args.num_queries, args.threshold, args.outdir, metrics)
    else:
        print("[seeds] skipped: no track_seed_prob in the prediction file (not a dynamic-query model, or written before it)")
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "queries_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    text = summary(metrics)
    with open(os.path.join(args.outdir, "queries_summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
