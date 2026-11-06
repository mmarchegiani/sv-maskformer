#!/usr/bin/env python3
"""Fit the normalization parameters (scale dicts) of the SV MaskFormer: the track input features and the vertex
regression targets, from converted parquet files.

    maskformer-sv norm-par --input <parquet file or directory> [--output params/sv_track_var_transform.yaml]
                           [--vertex-output params/sv_vertex_var_transform.yaml] [--max-events N] [--dry-run]

Two YAML files in the hepattn `FeatureScaler` format (see params/README.md), one per collection:

  --output          fitted on the derived TRACK features (sv.dataset.features.TRACK_FEATURES): every feature the
                    dataset can build gets an entry, so a config may pick any subset under `inputs.track`;
  --vertex-output   fitted on the derived VERTEX targets (VERTEX_TARGETS), the dict handed to the regression task and
                    the prediction writer (`particle_scale_dict_path` of the data block, aliased in the model block).

For each feature the derivation of sv.dataset.features is applied, then the entry's `fn` (vertex targets with `log`
/ `log1p`), and the statistics of the result are written: `type: std` entries carry the mean and the standard
deviation, `type: min_max_sym` entries the extremes (so the feature maps onto [-1, 1]), `identity` features
(sin/cos phi, flags, charge) mean 0 / std 1. The quantiles 0.1% and 99.9% are recorded as `min`/`max` of the `std`
entries for information. The dataset refuses non-finite values and values outside the domain of `fn`, so the fit
checks the same (`log` of a non-positive pT would fail here first).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import awkward as ak
import numpy as np
import yaml

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from maskformer.utils.parquet import list_parquet_files, read_parquet_files  # noqa: E402
from sv.dataset.collections import PV_FIELDS, TRACK_COLLECTION, VERTEX_COLLECTION  # noqa: E402
from sv.dataset.features import PV_KEYS, TRACK_FEATURES, VERTEX_TARGETS, Feature, apply_fn, branches_needed, derive  # noqa: E402

DEFAULT_TRACK_OUTPUT = "params/sv_track_var_transform.yaml"
DEFAULT_VERTEX_OUTPUT = "params/sv_vertex_var_transform.yaml"
QUANTILES = (0.1, 99.9)


def load_tables(files, max_events: int | None):
    """Flat numpy tables of the raw branches needed by every feature: (track table, vertex table with PV columns)."""
    table = read_parquet_files(files, columns=[TRACK_COLLECTION, VERTEX_COLLECTION, *PV_FIELDS])
    df = ak.from_arrow(table)
    if max_events is not None:
        df = df[:max_events]
    tracks, vertices = df[TRACK_COLLECTION], df[VERTEX_COLLECTION]
    track_table = {b: ak.to_numpy(ak.flatten(tracks[b])) for b in branches_needed(TRACK_FEATURES, TRACK_FEATURES)}
    n_gv = ak.to_numpy(ak.num(vertices["GV_pt"], axis=1))
    vertex_table = {}
    for b in branches_needed(VERTEX_TARGETS, VERTEX_TARGETS):
        if b in PV_KEYS:
            vertex_table[b] = np.repeat(ak.to_numpy(df[b]).astype(np.float64), n_gv)
        else:
            vertex_table[b] = ak.to_numpy(ak.flatten(vertices[b]))
    return len(df), track_table, vertex_table


def fit_entry(feature: Feature, values: np.ndarray) -> dict:
    """The scale-dict entry of one feature from its derived values."""
    if not np.isfinite(values).all():
        raise AssertionError(f"{feature.name}: {int((~np.isfinite(values)).sum())} non-finite derived values")
    transformed = apply_fn(feature.fn, values)
    if not np.isfinite(transformed).all():
        raise AssertionError(f"{feature.name}: values outside the domain of fn={feature.fn} (min {values.min():.4g})")
    lo, hi = (float(q) for q in np.percentile(transformed, QUANTILES))
    entry: dict = {"type": "std" if feature.scaling == "identity" else feature.scaling}
    if feature.fn:
        entry["fn"] = feature.fn
    if feature.scaling == "identity":
        entry.update(mean=0.0, std=1.0)
    elif feature.scaling == "std":
        std = float(transformed.std())
        if std == 0:
            raise AssertionError(f"{feature.name}: constant feature (std 0)")
        entry.update(mean=round(float(transformed.mean()), 6), std=round(std, 6), min=round(lo, 6), max=round(hi, 6))
    elif feature.scaling == "min_max_sym":
        vmin, vmax = float(transformed.min()), float(transformed.max())
        if vmax == vmin:
            raise AssertionError(f"{feature.name}: constant feature")
        entry.update(mean=round(float(transformed.mean()), 6), std=round(float(transformed.std()), 6),
                     min=round(vmin, 6), max=round(vmax, 6))
    else:
        raise ValueError(f"{feature.name}: unknown scaling {feature.scaling!r}")
    return entry


def fit(features: dict[str, Feature], table: dict) -> dict[str, dict]:
    derived = derive(features, list(features), table)
    return {name: fit_entry(features[name], derived[name]) for name in features}


def report(label: str, features: dict[str, Feature], entries: dict[str, dict], n: int) -> None:
    print(f"{label}: {n:,} entries")
    print(f"  {'feature':12s} {'type':12s} {'fn':6s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}  comment")
    for name, e in entries.items():
        print(f"  {name:12s} {e['type']:12s} {str(e.get('fn') or '-'):6s} {e.get('mean', 0.0):10.4f} {e.get('std', 1.0):10.4f} "
              f"{e.get('min', float('nan')):10.4f} {e.get('max', float('nan')):10.4f}  {features[name].comment}")


def to_yaml(entries: dict[str, dict], features: dict[str, Feature], files, n_events: int, n_entries: int, role: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Generated by sv/scripts/dataset/compute_norm_par.py on {stamp}",
        f"# {role}",
        f"# input: {', '.join(str(f) for f in files)} ({n_events:,} events, {n_entries:,} entries)",
        "# NOTE: mean/std/min/max are statistics of the DERIVED variable (sv.dataset.features) after `fn`;",
        "#       hepattn's VarTransform applies `fn` first, then (x - shift)/scale. min/max of `std` entries are",
        f"#       the {QUANTILES[0]}% / {QUANTILES[1]}% quantiles, for information.",
    ]
    for name, entry in entries.items():
        lines.append(f"{name}:  # {features[name].comment}")
        for key, value in entry.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Converted parquet file, or a directory of them")
    p.add_argument("--output", default=None, help=f"Track scale dict (default: {DEFAULT_TRACK_OUTPUT} under the repository)")
    p.add_argument("--vertex-output", default=None, help=f"Vertex scale dict (default: {DEFAULT_VERTEX_OUTPUT} under the repository)")
    p.add_argument("--max-files", type=int, default=None, help="Only read the first N files of the directory")
    p.add_argument("--max-events", type=int, default=None, help="Only use the first N events")
    p.add_argument("--dry-run", action="store_true", help="Print the YAML instead of writing it")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    files = list_parquet_files(args.input)
    if args.max_files:
        files = files[: args.max_files]
    repo = SRC_DIR.parent
    track_out = Path(args.output) if args.output else repo / DEFAULT_TRACK_OUTPUT
    vertex_out = Path(args.vertex_output) if args.vertex_output else repo / DEFAULT_VERTEX_OUTPUT

    n_events, track_table, vertex_table = load_tables(files, args.max_events)
    n_tracks = len(next(iter(track_table.values())))
    n_vertices = len(next(iter(vertex_table.values())))
    track_entries = fit(TRACK_FEATURES, track_table)
    vertex_entries = fit(VERTEX_TARGETS, vertex_table)
    report(f"{TRACK_COLLECTION} features", TRACK_FEATURES, track_entries, n_tracks)
    report(f"{VERTEX_COLLECTION} targets", VERTEX_TARGETS, vertex_entries, n_vertices)

    outputs = [
        (track_out, to_yaml(track_entries, TRACK_FEATURES, files, n_events, n_tracks, f"collection: {TRACK_COLLECTION} (input features)")),
        (vertex_out, to_yaml(vertex_entries, VERTEX_TARGETS, files, n_events, n_vertices, f"collection: {VERTEX_COLLECTION} (regression targets)")),
    ]
    for path, text in outputs:
        if args.dry_run:
            print(f"--- {path} ---\n{text}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
