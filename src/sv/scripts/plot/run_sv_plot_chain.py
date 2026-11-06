#!/usr/bin/env python
"""The SV plot chain on one prediction file (what `maskformer-sv evaluate` runs after the test loop).

Loads the file once (``sv_eval_common.EvalData``) and runs the sections of ``plot_sv_eval`` (efficiency/, fakes/,
classification/, regression/, tracks/) and of ``plot_sv_resolution`` (resolution/: the absolute and relative
residual of every regressed property), then writes metrics.json and summary.txt. A failing section is reported
(and recorded in ``<outdir>/plot_chain_failures.txt``) without stopping the others. Needs awkward, numpy, pyarrow
and matplotlib (mplhep optional), no torch.

    maskformer-sv plot --input <ckpt_dir>/<ckpt>__test.parquet --outdir <ckpt_dir>/<ckpt>__test_plots
    python src/sv/scripts/plot/run_sv_plot_chain.py --input <file> --outdir <dir> [--max-events N]

Exit code: 0 all sections done, 1 some section failed, 2 the file could not be loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sv_eval_common import EvalData  # noqa: E402
import plot_sv_eval as ev  # noqa: E402
import plot_sv_resolution as res  # noqa: E402

SECTIONS = ("efficiency", "fakes", "classification", "regression", "tracks", "resolution")


def section(name: str):
    """The function of a section: ``<name>_section(data, outdir, metrics)``."""
    return res.resolution_section if name == "resolution" else getattr(ev, f"{name}_section")


def run_chain(input, outdir, track_collection="track", vertex_collection="GV", max_events=None) -> dict:
    """Run every section; returns ``dict(metrics, failures, outdir)``. Raises if the file cannot be loaded."""
    print(f"[chain] loading {input}", flush=True)
    data = EvalData(input, max_events=max_events, track_collection=track_collection, vertex_collection=vertex_collection)
    print(f"[chain] {data.summary()}", flush=True)
    os.makedirs(outdir, exist_ok=True)
    metrics: dict = {"input": str(input), "n_events": data.n_events, "n_vertices": data.n_vertices, "eval_min_tracks": ev.EVAL_MIN_TRACKS}
    failures: dict[str, str] = {}
    for name in SECTIONS:
        print(f"[chain] {name} ...", flush=True)
        try:
            section(name)(data, outdir, metrics)
        except Exception:  # noqa: BLE001 - one broken section must not hide the others
            failures[name] = traceback.format_exc()
            print(f"[chain] {name} FAILED:\n{failures[name]}", flush=True)
            metrics.setdefault(name, {})
    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    try:
        ev.write_summary(data, metrics, outdir)
    except Exception:  # noqa: BLE001
        failures["summary"] = traceback.format_exc()
        print(f"[chain] summary FAILED:\n{failures['summary']}", flush=True)
    if failures:
        with open(os.path.join(outdir, "plot_chain_failures.txt"), "w") as fh:
            for name, tb in failures.items():
                fh.write(f"==== {name}\n{tb}\n")
        print(f"[chain] {len(failures)} section(s) failed: {sorted(failures)}", flush=True)
    else:
        print(f"[chain] done -> {outdir}", flush=True)
    return {"metrics": metrics, "failures": failures, "outdir": outdir}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="prediction parquet file")
    p.add_argument("--outdir", required=True, help="output directory")
    p.add_argument("--track-collection", default="track")
    p.add_argument("--vertex-collection", default="GV")
    p.add_argument("--max-events", type=int, default=None)
    args = p.parse_args(argv)
    try:
        result = run_chain(args.input, args.outdir, args.track_collection, args.vertex_collection, args.max_events)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"[chain] could not load {args.input}")
        return 2
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
