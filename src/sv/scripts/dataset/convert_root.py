#!/usr/bin/env python3
"""Convert BTV secondary-vertex ntuples (ROOT, `Events` tree) to the parquet layout of the SV dataset.

    maskformer-sv convert /eos/user/m/mmarcheg/BTV/sv_reconstruction/ttbar_BTV_1.root --outdir <dir>
    maskformer-sv convert <files or globs>... --outdir <dir> --split 0.8 0.1 0.1      # train/ val/ test/ sub-folders
    python src/sv/scripts/dataset/convert_root.py <file.root> --outdir <dir>           # the same without the CLI

Each input file becomes `<outdir>/<stem>.parquet` (or, with `--split`, a share of its events in each of
`<outdir>/train/<stem>.parquet`, `<outdir>/val/<stem>.parquet` and `<outdir>/test/<stem>.parquet`, cut in file order,
so the splits of several files can be given to the data module as directories). One record per event:

    run, luminosityBlock, event              event identifiers
    PV_x, PV_y, PV_z                         the generated primary vertex (vertex of the first status-21 GenPart;
                                             fallback: the production vertex of a prompt vertex, then 0)
    track                                    struct of the track branches (sv.dataset.collections.TRACK_BRANCHES)
    GV                                       struct of the vertex branches plus the derived truth fields
                                             GV_class, GV_category, GV_nTracks, GV_lxy, GV_lz, GV_l3d (from the PV),
                                             GV_flight_xy, GV_flight_3d (from the hadron's production vertex)
    GVDaughters, RejectedGV, trackPair       optional (--keep), unchanged

Needs uproot, awkward and pyarrow only (no torch). The file is read in chunks of `--step` events.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import uproot

SRC_DIR = Path(__file__).resolve().parents[3]  # src/, for runs by file path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sv.dataset.collections import (  # noqa: E402
    EVENT_BRANCHES,
    GENPART_HARD_STATUS,
    GENPART_PV_BRANCHES,
    OPTIONAL_COLLECTIONS,
    PV_FIELDS,
    TRACK_BRANCHES,
    TRACK_COLLECTION,
    TRACK_TRUTH_IDX,
    VERTEX_BRANCHES,
    VERTEX_COLLECTION,
    VERTEX_DERIVED,
    vertex_category,
    vertex_class,
)

TREE = "Events"
SPLITS = ("train", "val", "test")


def gen_primary_vertex(chunk: ak.Array) -> dict[str, ak.Array]:
    """Per-event (x, y, z) of the generated primary vertex: the first GenPart with the hard-scatter status, else the
    production vertex of the first prompt vertex (`GV_fromHF == 0`), else 0."""
    hard = chunk["GenPart_status"] == GENPART_HARD_STATUS
    prompt = chunk["GV_fromHF"] == 0
    out = {}
    for axis in "xyz":
        # numpy with NaN for "none": ak.fill_none would mix dtypes into a union type
        from_genpart = ak.to_numpy(ak.fill_none(ak.values_astype(ak.firsts(chunk[f"GenPart_v{axis}"][hard]), np.float64), np.nan))
        from_gv = ak.to_numpy(ak.fill_none(ak.values_astype(ak.firsts(chunk[f"GV_{axis}_i"][prompt]), np.float64), np.nan))
        value = np.where(np.isnan(from_genpart), np.where(np.isnan(from_gv), 0.0, from_gv), from_genpart)
        out[f"PV_{axis}"] = ak.Array(value.astype(np.float32))
    return out


def vertex_truth_fields(chunk: ak.Array, pv: dict[str, ak.Array]) -> dict[str, ak.Array]:
    """The derived truth fields of the vertex collection (VERTEX_DERIVED), jagged per event."""
    n_gv = ak.num(chunk["GV_pt"], axis=1)
    counts = ak.to_numpy(n_gv)
    flat = lambda name: ak.to_numpy(ak.flatten(chunk[name]))  # noqa: E731
    cls = vertex_class(flat("GV_isB"), flat("GV_isD"))
    cat = vertex_category(flat("GV_isB"), flat("GV_isD"), flat("GV_fromHF"))

    # tracks pointing to each vertex (track_hadidx indexes the event's GV list)
    hadidx = chunk[TRACK_TRUTH_IDX]
    n_tracks = ak.to_numpy(ak.num(hadidx, axis=1))
    flat_hadidx = ak.to_numpy(ak.flatten(hadidx))
    gv_offsets = np.concatenate([[0], np.cumsum(counts)])
    track_event = np.repeat(np.arange(len(counts)), n_tracks)
    valid = flat_hadidx >= 0
    if valid.any() and (flat_hadidx[valid] >= counts[track_event[valid]]).any():
        raise ValueError(f"{TRACK_TRUTH_IDX} points outside the {VERTEX_COLLECTION} collection")
    n_per_vertex = np.bincount(gv_offsets[track_event[valid]] + flat_hadidx[valid], minlength=int(counts.sum()))

    pv_per_vertex = {axis: np.repeat(ak.to_numpy(pv[f"PV_{axis}"]).astype(np.float64), counts) for axis in "xyz"}
    x, y, z = (flat(f"GV_{a}").astype(np.float64) for a in "xyz")
    xi, yi, zi = (flat(f"GV_{a}_i").astype(np.float64) for a in "xyz")
    lxy = np.hypot(x - pv_per_vertex["x"], y - pv_per_vertex["y"])
    lz = z - pv_per_vertex["z"]
    flight_xy = np.hypot(x - xi, y - yi)
    flight_3d = np.sqrt(flight_xy**2 + (z - zi) ** 2)

    def jag(values, dtype):
        return ak.unflatten(np.asarray(values, dtype=dtype), counts)

    return {
        "GV_class": jag(cls, np.int32),
        "GV_category": jag(cat, np.int32),
        "GV_nTracks": jag(n_per_vertex, np.int32),
        "GV_lxy": jag(lxy, np.float32),
        "GV_lz": jag(lz, np.float32),
        "GV_l3d": jag(np.sqrt(lxy**2 + lz**2), np.float32),
        "GV_flight_xy": jag(flight_xy, np.float32),
        "GV_flight_3d": jag(flight_3d, np.float32),
    }


def convert_chunk(chunk: ak.Array, keep: tuple[str, ...] = ()) -> ak.Array:
    """One event record per entry of `chunk` (an awkward array of the raw branches)."""
    pv = gen_primary_vertex(chunk)
    fields: dict[str, ak.Array] = {name: chunk[name] for name in EVENT_BRANCHES}
    fields.update(pv)
    fields[TRACK_COLLECTION] = ak.zip({name: chunk[name] for name in TRACK_BRANCHES}, depth_limit=1)
    gv = {name: chunk[name] for name in VERTEX_BRANCHES}
    gv.update(vertex_truth_fields(chunk, pv))
    fields[VERTEX_COLLECTION] = ak.zip(gv, depth_limit=1)
    for collection in keep:
        fields[collection] = ak.zip({name: chunk[name] for name in OPTIONAL_COLLECTIONS[collection]}, depth_limit=1)
    return ak.zip(fields, depth_limit=1)


def branches_to_read(keep: tuple[str, ...] = ()) -> list[str]:
    names = [*EVENT_BRANCHES, *GENPART_PV_BRANCHES, *TRACK_BRANCHES, *VERTEX_BRANCHES]
    for collection in keep:
        names += list(OPTIONAL_COLLECTIONS[collection])
    return names


def convert_file(path: str, step: int = 1000, max_events: int | None = None, keep: tuple[str, ...] = ()) -> ak.Array:
    """All events of the ROOT file `path` as one awkward array of event records."""
    chunks = []
    n_done = 0
    with uproot.open(path) as f:
        tree = f[TREE]
        n_total = tree.num_entries if max_events is None else min(tree.num_entries, max_events)
        missing = [b for b in branches_to_read(keep) if b not in tree]
        if missing:
            raise ValueError(f"{path}:{TREE} lacks the branches {missing}")
        for chunk in tree.iterate(branches_to_read(keep), step_size=step, entry_stop=n_total, library="ak"):
            chunks.append(convert_chunk(chunk, keep))
            n_done += len(chunk)
            print(f"  {path}: {n_done}/{n_total} events", flush=True)
    if not chunks:
        raise ValueError(f"{path}:{TREE} has no events")
    return ak.concatenate(chunks)


def split_counts(n: int, fractions: list[float]) -> list[int]:
    """Events per split for `n` events and the three fractions (the last split takes the remainder)."""
    if len(fractions) != len(SPLITS) or any(f < 0 for f in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"--split needs {len(SPLITS)} non-negative fractions summing to 1, got {fractions}")
    counts = [int(round(f * n)) for f in fractions[:-1]]
    counts.append(n - sum(counts))
    if min(counts) < 0:
        raise ValueError(f"split {fractions} of {n} events gives {counts}")
    return counts


def to_parquet(events: ak.Array, path: Path) -> None:
    """Write the event records through pyarrow (plain arrow types, no awkward extension types): `ak.to_parquet`
    fails on some awkward/pyarrow combinations ("file metadata is only available after writer close")."""
    pq.write_table(ak.to_arrow_table(events, extensionarray=False), path)


def write(events: ak.Array, outdir: Path, stem: str, split: list[float] | None) -> list[Path]:
    written = []
    if split is None:
        outdir.mkdir(parents=True, exist_ok=True)
        path = outdir / f"{stem}.parquet"
        to_parquet(events, path)
        written.append(path)
        print(f"wrote {len(events)} events to {path}")
        return written
    counts = split_counts(len(events), split)
    start = 0
    for name, count in zip(SPLITS, counts):
        part = events[start : start + count]
        start += count
        if count == 0:
            print(f"{name}: no events (fraction 0), nothing written")
            continue
        (outdir / name).mkdir(parents=True, exist_ok=True)
        path = outdir / name / f"{stem}.parquet"
        to_parquet(part, path)
        written.append(path)
        print(f"wrote {count} events to {path}")
    return written


def expand_inputs(patterns) -> list[str]:
    files = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        if not matches:
            raise FileNotFoundError(f"no file matches {pattern}")
        files += matches
    for f in files:
        if not Path(f).is_file():
            raise FileNotFoundError(f)
    return files


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="ROOT files (or globs) with the Events tree")
    p.add_argument("--outdir", required=True, help="Output directory (created)")
    p.add_argument("--split", nargs=3, type=float, default=None, metavar=("TRAIN", "VAL", "TEST"),
                   help="Cut the events of every file into train/ val/ test/ sub-folders by these fractions (file order)")
    p.add_argument("--max-events", type=int, default=None, help="Only convert the first N events of every file")
    p.add_argument("--step", type=int, default=1000, help="Events read per chunk (default 1000)")
    p.add_argument("--keep", action="append", choices=sorted(OPTIONAL_COLLECTIONS), default=[],
                   help="Also keep this optional collection (repeatable): " + ", ".join(sorted(OPTIONAL_COLLECTIONS)))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    files = expand_inputs(args.inputs)
    outdir = Path(args.outdir)
    keep = tuple(dict.fromkeys(args.keep))
    print(f"converting {len(files)} file(s) to {outdir}" + (f", split {args.split}" if args.split else ""))
    for path in files:
        events = convert_file(path, step=args.step, max_events=args.max_events, keep=keep)
        n_gv = ak.to_numpy(ak.num(events[VERTEX_COLLECTION]["GV_pt"], axis=1))
        n_tr = ak.to_numpy(ak.num(events[TRACK_COLLECTION]["track_pt"], axis=1))
        print(f"  {Path(path).name}: {len(events)} events, {int(n_tr.sum())} tracks (max {int(n_tr.max())}/event), "
              f"{int(n_gv.sum())} vertices (max {int(n_gv.max())}/event), derived fields {list(VERTEX_DERIVED)}, PV {list(PV_FIELDS)}")
        write(events, outdir, Path(path).stem, args.split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
