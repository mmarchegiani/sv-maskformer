#!/usr/bin/env python3
"""Convert BTV secondary-vertex ntuples (ROOT, `Events` tree) to the parquet layout of the SV dataset.

    maskformer-sv convert /eos/user/m/mmarcheg/BTV/sv_reconstruction/ttbar_BTV_1.root --outdir <dir>
    maskformer-sv convert <files or globs>... --outdir <dir> --split 0.8 0.1 0.1      # train/ val/ test/ sub-folders
    python src/sv/scripts/dataset/convert_root.py <file.root> --outdir <dir>           # the same without the CLI

Each input file becomes `<outdir>/<stem>.parquet` (or, with `--split`, a share of its events in each of
`<outdir>/train/<stem>.parquet`, `<outdir>/val/<stem>.parquet` and `<outdir>/test/<stem>.parquet`, cut in file order,
so the splits of several files can be given to the data module as directories). One record per event:

    run, luminosityBlock, event              event identifiers
    PV_x, PV_y, PV_z                         the generated primary vertex (vertex of the first status-21 GenPart,
                                             or customGenPart in the ntuples that name the collection so;
                                             fallback: the production vertex of a prompt vertex, then 0)
    track                                    struct of the track branches (sv.dataset.collections.TRACK_BRANCHES)
    GV                                       struct of the vertex branches plus the derived truth fields
                                             GV_class, GV_nTracks, GV_nSelectedDaughters, GV_lxy, GV_lz,
                                             GV_l3d (from the PV), GV_flight_xy, GV_flight_3d (from the hadron's
                                             production vertex)
    GVDaughters, RejectedGV, trackPair       optional (--keep)

The candidate vertices are selected (sv.dataset.collections.VERTEX_SELECTION): hadron pT > 10 GeV, |eta| < 2.5 and
at least two daughters of the `GVDaughters` collection that are charged, of a generator-stable species, with
pT > 0.8 GeV and |eta| < 2.5 (DAUGHTER_SELECTION; their count is `GV_nSelectedDaughters`). The vertices failing it
are dropped from `GV`, the `track_hadidx` of their tracks is reset to -1 (and `track_label` to 0), the indices of the
remaining vertices are renumbered; a kept `GVDaughters` loses the daughters of the dropped vertices (`hadronIndex`
renumbered) and a kept `trackPair` gets its `edgeLabel` recomputed ("same `track_hadidx` >= 0"). The cutflow is
printed per file. The `GVDaughters` branches of the selection are therefore always needed.

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
    DAUGHTER_BRANCHES,
    DAUGHTER_COLLECTION,
    EVENT_BRANCHES,
    GENPART_COLLECTIONS,
    GENPART_HARD_STATUS,
    GENPART_PV_BRANCHES,
    OPTIONAL_COLLECTIONS,
    PV_FIELDS,
    TRACK_BRANCHES,
    TRACK_COLLECTION,
    TRACK_TRUTH_IDX,
    TRACK_TRUTH_LABEL,
    VERTEX_BRANCHES,
    VERTEX_COLLECTION,
    VERTEX_DERIVED,
    select_daughters,
    vertex_class,
    vertex_selection_flags,
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


def _check_index(name: str, index: np.ndarray, counts_per_event: np.ndarray, event: np.ndarray, allow_none: bool) -> np.ndarray:
    """Mask of the valid entries of a per-object index into another per-event collection (`counts_per_event`
    objects per event); -1 means "none" when `allow_none`. Raises on an index outside the collection."""
    valid = index >= 0 if allow_none else np.ones(index.shape, dtype=bool)
    if (index[~valid] < -1).any() or (valid.any() and (index[valid] >= counts_per_event[event[valid]]).any()):
        raise ValueError(f"{name} points outside the {VERTEX_COLLECTION} collection")
    return valid


def apply_vertex_selection(chunk: ak.Array, keep: tuple[str, ...] = (), cutflow: dict[str, int] | None = None) -> dict[str, ak.Array]:
    """The candidate vertex selection (collections.VERTEX_SELECTION / DAUGHTER_SELECTION) on a chunk of raw branches:
    the branches that change (jagged per event), to override the chunk's. The `GV_*` branches lose the failing
    vertices and gain `GV_nSelectedDaughters`; `track_hadidx` is renumbered (-1, and `track_label` 0, for the tracks of
    a dropped vertex); the kept optional collections follow (`GVDaughters` loses the daughters of the dropped
    vertices and `hadronIndex` is renumbered, `trackPair_edgeLabel` is recomputed). `cutflow` accumulates the counts."""
    flat = lambda name: ak.to_numpy(ak.flatten(chunk[name]))  # noqa: E731
    counts = ak.to_numpy(ak.num(chunk["GV_pt"], axis=1)).astype(np.int64)
    n_events, n_gv = len(counts), int(counts.sum())
    gv_offsets = np.concatenate([[0], np.cumsum(counts)])
    gv_event = np.repeat(np.arange(n_events), counts)

    # daughters passing the daughter selection, counted per vertex
    d_counts = ak.to_numpy(ak.num(chunk["GVDaughters_hadronIndex"], axis=1)).astype(np.int64)
    d_event = np.repeat(np.arange(n_events), d_counts)
    d_index = flat("GVDaughters_hadronIndex").astype(np.int64)
    _check_index("GVDaughters_hadronIndex", d_index, counts, d_event, allow_none=False)
    d_pass = select_daughters(flat("GVDaughters_charge"), flat("GVDaughters_pt"), flat("GVDaughters_eta"), flat("GVDaughters_pdgId"))
    n_selected = np.bincount(gv_offsets[d_event[d_pass]] + d_index[d_pass], minlength=n_gv).astype(np.int64)

    flags = vertex_selection_flags(flat("GV_pt"), flat("GV_eta"), n_selected)
    passing = np.ones(n_gv, dtype=bool)
    if cutflow is not None:
        cutflow["vertices"] = cutflow.get("vertices", 0) + n_gv
    for name, flag in flags.items():
        passing &= flag
        if cutflow is not None:
            cutflow[name] = cutflow.get(name, 0) + int(passing.sum())
    kept_per_event = np.bincount(gv_event, weights=passing, minlength=n_events).astype(np.int64)
    new_offsets = np.concatenate([[0], np.cumsum(kept_per_event)])
    new_local = np.where(passing, np.cumsum(passing) - 1 - new_offsets[gv_event], -1)  # new index within the event

    out: dict[str, ak.Array] = {}
    gv_mask = ak.unflatten(passing, counts)
    for name in VERTEX_BRANCHES:
        out[name] = chunk[name][gv_mask]
    out["GV_nSelectedDaughters"] = ak.unflatten(n_selected[passing].astype(np.int32), kept_per_event)

    # tracks: renumber the truth index, reset the tracks of the dropped vertices
    hadidx = flat(TRACK_TRUTH_IDX)
    t_counts = ak.to_numpy(ak.num(chunk[TRACK_TRUTH_IDX], axis=1)).astype(np.int64)
    t_event = np.repeat(np.arange(n_events), t_counts)
    valid = _check_index(TRACK_TRUTH_IDX, hadidx.astype(np.int64), counts, t_event, allow_none=True)
    new_hadidx = np.full(hadidx.shape, -1, dtype=np.int64)
    new_hadidx[valid] = new_local[gv_offsets[t_event[valid]] + hadidx[valid].astype(np.int64)]
    reset = valid & (new_hadidx < 0)
    label = flat(TRACK_TRUTH_LABEL).copy()
    label[reset] = 0
    if cutflow is not None:
        cutflow["tracks reset to no vertex"] = cutflow.get("tracks reset to no vertex", 0) + int(reset.sum())
    out[TRACK_TRUTH_IDX] = ak.unflatten(new_hadidx.astype(hadidx.dtype), t_counts)
    out[TRACK_TRUTH_LABEL] = ak.unflatten(label, t_counts)

    if DAUGHTER_COLLECTION in keep:
        d_new = new_local[gv_offsets[d_event] + d_index]
        d_keep = d_new >= 0
        d_mask = ak.unflatten(d_keep, d_counts)
        for name in OPTIONAL_COLLECTIONS[DAUGHTER_COLLECTION]:
            out[name] = chunk[name][d_mask]
        out["GVDaughters_hadronIndex"] = ak.unflatten(d_new[d_keep].astype(d_index.dtype), np.bincount(d_event[d_keep], minlength=n_events))
    if "trackPair" in keep:
        p_counts = ak.to_numpy(ak.num(chunk["trackPair_trk1Idx"], axis=1)).astype(np.int64)
        t_offsets = np.concatenate([[0], np.cumsum(t_counts)])[np.repeat(np.arange(n_events), p_counts)]
        h1 = new_hadidx[t_offsets + flat("trackPair_trk1Idx").astype(np.int64)]
        h2 = new_hadidx[t_offsets + flat("trackPair_trk2Idx").astype(np.int64)]
        out["trackPair_edgeLabel"] = ak.unflatten(((h1 == h2) & (h1 >= 0)).astype(flat("trackPair_edgeLabel").dtype), p_counts)
    return out


def vertex_truth_fields(chunk, pv: dict[str, ak.Array]) -> dict[str, ak.Array]:
    """The derived truth fields of the vertex collection (VERTEX_DERIVED but `GV_nSelectedDaughters`, which the
    selection computes), jagged per event; `chunk` maps the branch names to jagged arrays after the selection."""
    n_gv = ak.num(chunk["GV_pt"], axis=1)
    counts = ak.to_numpy(n_gv)
    flat = lambda name: ak.to_numpy(ak.flatten(chunk[name]))  # noqa: E731
    cls = vertex_class(flat("GV_isB"), flat("GV_isD"), flat("GV_fromHF"))

    # tracks pointing to each vertex (track_hadidx indexes the event's GV list)
    hadidx = chunk[TRACK_TRUTH_IDX]
    n_tracks = ak.to_numpy(ak.num(hadidx, axis=1))
    flat_hadidx = ak.to_numpy(ak.flatten(hadidx)).astype(np.int64)
    gv_offsets = np.concatenate([[0], np.cumsum(counts)])
    track_event = np.repeat(np.arange(len(counts)), n_tracks)
    valid = _check_index(TRACK_TRUTH_IDX, flat_hadidx, counts, track_event, allow_none=True)
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
        "GV_nTracks": jag(n_per_vertex, np.int32),
        "GV_lxy": jag(lxy, np.float32),
        "GV_lz": jag(lz, np.float32),
        "GV_l3d": jag(np.sqrt(lxy**2 + lz**2), np.float32),
        "GV_flight_xy": jag(flight_xy, np.float32),
        "GV_flight_3d": jag(flight_3d, np.float32),
    }


def convert_chunk(chunk: ak.Array, keep: tuple[str, ...] = (), cutflow: dict[str, int] | None = None) -> ak.Array:
    """One event record per entry of `chunk` (an awkward array of the raw branches), after the candidate vertex
    selection."""
    branches = {name: chunk[name] for name in chunk.fields}
    branches.update(apply_vertex_selection(chunk, keep, cutflow))
    pv = gen_primary_vertex(branches)
    fields: dict[str, ak.Array] = {name: branches[name] for name in EVENT_BRANCHES}
    fields.update(pv)
    fields[TRACK_COLLECTION] = ak.zip({name: branches[name] for name in TRACK_BRANCHES}, depth_limit=1)
    gv = {name: branches[name] for name in VERTEX_BRANCHES}
    gv["GV_nSelectedDaughters"] = branches["GV_nSelectedDaughters"]
    gv.update(vertex_truth_fields(branches, pv))
    fields[VERTEX_COLLECTION] = ak.zip(gv, depth_limit=1)
    for collection in keep:
        fields[collection] = ak.zip({name: branches[name] for name in OPTIONAL_COLLECTIONS[collection]}, depth_limit=1)
    return ak.zip(fields, depth_limit=1)


def genpart_aliases(available) -> dict[str, str]:
    """Branch of the file -> GENPART_PV_BRANCHES name, for the first generator-particle collection of
    GENPART_COLLECTIONS the file has completely (empty when it is `GenPart` itself or none is there)."""
    for collection in GENPART_COLLECTIONS:
        names = {b.replace("GenPart", collection, 1): b for b in GENPART_PV_BRANCHES}
        if all(name in available for name in names):
            return {name: b for name, b in names.items() if name != b}
    return {}


def branches_to_read(keep: tuple[str, ...] = ()) -> list[str]:
    names = [*EVENT_BRANCHES, *GENPART_PV_BRANCHES, *TRACK_BRANCHES, *VERTEX_BRANCHES, *DAUGHTER_BRANCHES]
    for collection in keep:
        names += list(OPTIONAL_COLLECTIONS[collection])
    return list(dict.fromkeys(names))


def convert_file(path: str, step: int = 1000, max_events: int | None = None, keep: tuple[str, ...] = ()) -> ak.Array:
    """All events of the ROOT file `path` as one awkward array of event records."""
    chunks = []
    n_done = 0
    with uproot.open(path) as f:
        tree = f[TREE]
        n_total = tree.num_entries if max_events is None else min(tree.num_entries, max_events)
        aliases = genpart_aliases(tree.keys())  # file branch -> canonical GenPart name
        to_file = {canonical: name for name, canonical in aliases.items()}
        names = [to_file.get(b, b) for b in branches_to_read(keep)]
        missing = [b for b in names if b not in tree]
        if missing:
            note = " (needed for the candidate vertex selection)" if any(b in DAUGHTER_BRANCHES for b in missing) else ""
            raise ValueError(f"{path}:{TREE} lacks the branches {missing}{note}")
        cutflow: dict[str, int] = {}
        for chunk in tree.iterate(names, step_size=step, entry_stop=n_total, library="ak"):
            if aliases:
                chunk = ak.zip({aliases.get(name, name): chunk[name] for name in chunk.fields}, depth_limit=1)
            chunks.append(convert_chunk(chunk, keep, cutflow))
            n_done += len(chunk)
            print(f"  {path}: {n_done}/{n_total} events", flush=True)
    if not chunks:
        raise ValueError(f"{path}:{TREE} has no events")
    print(f"  {path}: candidate vertex selection cutflow {cutflow}")
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
