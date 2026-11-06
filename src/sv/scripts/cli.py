"""`maskformer-sv`: the command line of the SV (secondary-vertex) MaskFormer.

    maskformer-sv convert   <ntuple.root>... --outdir <dir> [--split 0.8 0.1 0.1]     ROOT -> parquet
    maskformer-sv norm-par  --input <parquet dir or file>                              fit the two scale dicts into params/
    maskformer-sv train     --config configs/sv/gv_tracks.yaml [Lightning arguments]
    maskformer-sv evaluate  --config logs/<run>/config.yaml --test-path <file.parquet> [Lightning arguments]
    maskformer-sv plot      --input <ckpt>__test.parquet --outdir <dir>                the plot chain on a prediction file

`train` and `evaluate` are built from the generic pieces of `maskformer.scripts.cli` and bound to the SV entry points
(sv.scripts.train / sv.scripts.evaluate): the options cover the usual knobs, any other argument (`--trainer.max_epochs 5`,
`--data.num_workers=0`, `--model.model.init_args...`, `--print_config`) is forwarded unchanged.

Only click is imported at module level; torch, lightning and the scripts are imported inside the commands, so
`maskformer-sv --help`, `convert`, `norm-par` and `plot` work in a torch-free environment (e.g. an LCG view). Without
the package installed: `PYTHONPATH=src python -m sv ...` from the repository root.
"""

from __future__ import annotations

import sys
import traceback

import click

from maskformer.scripts.cli import (
    LIGHTNING_EPILOG,
    PASSTHROUGH,
    _extra,
    _opt,
    evaluate_argv,
    evaluate_options,
    extra_args_argument,
    lightning_help_option,
    make_train_command,
)


def _train_main():
    from sv.scripts.train import main
    return main


def _evaluate_main():
    from sv.scripts.evaluate import main
    return main


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    """SV MaskFormer: secondary-vertex reconstruction from tracks (conversion, normalization, training, evaluation, plots)."""


cli.add_command(make_train_command(_train_main, help="Train the SV MaskFormer. The config alone selects the model and the data."))


# -------------------------------------------------------------------------------------------- evaluate


@cli.command(epilog=LIGHTNING_EPILOG, **PASSTHROUGH)
@evaluate_options
@click.option("--plot-dir", default=None, help="Directory of the evaluation plots (--plot_dir); default: <ckpt_dir>/<ckpt>__test[_<suff>]_plots/.")
@click.option("--plot-max-events", type=int, default=None, help="Plot only the first N events (--plot_max_events).")
@lightning_help_option("test", _evaluate_main)
@extra_args_argument
def evaluate(plot_dir, plot_max_events, extra_args, **options):
    """Evaluate a trained SV MaskFormer: predictions and losses next to the checkpoint, then the plot chain."""
    argv = evaluate_argv(**options)
    argv += _opt("--plot_dir", plot_dir)
    argv += _opt("--plot_max_events", plot_max_events)
    argv += _extra(extra_args)
    _evaluate_main()(argv)


# --------------------------------------------------------------------------------------------- convert


@cli.command()
@click.argument("inputs", nargs=-1, required=True)
@click.option("--outdir", required=True, help="Output directory of the parquet files.")
@click.option("--split", nargs=3, type=float, default=None, metavar="TRAIN VAL TEST",
              help="Cut every file's events into train/ val/ test/ sub-folders by these fractions.")
@click.option("--max-events", type=int, default=None, help="Only convert the first N events of every file.")
@click.option("--step", type=int, default=None, help="Events read per chunk (default 1000).")
@click.option("--keep", multiple=True, help="Also keep this optional collection (GVDaughters, RejectedGV, trackPair); repeatable.")
def convert(inputs, outdir, split, max_events, step, keep):
    """Convert BTV secondary-vertex ntuples (ROOT files or globs) to the parquet layout of the SV dataset (torch-free)."""
    from sv.scripts.dataset.convert_root import main

    argv = [*inputs, "--outdir", outdir]
    argv += _opt("--max-events", max_events)
    if split is not None:
        argv += ["--split", *(str(s) for s in split)]
    argv += _opt("--step", step)
    for collection in keep:
        argv += ["--keep", collection]
    sys.exit(main(argv))


# -------------------------------------------------------------------------------------------- norm-par


@cli.command("norm-par")
@click.option("--input", "input_path", required=True, help="Converted parquet file, or a directory of them.")
@click.option("--output", default=None, help="Track scale dict (default: params/sv_track_var_transform.yaml).")
@click.option("--vertex-output", default=None, help="Vertex scale dict (default: params/sv_vertex_var_transform.yaml).")
@click.option("--max-files", type=int, default=None, help="Only read the first N files.")
@click.option("--max-events", type=int, default=None, help="Only use the first N events.")
@click.option("--dry-run", is_flag=True, help="Print the YAML instead of writing it.")
def norm_par(input_path, output, vertex_output, max_files, max_events, dry_run):
    """Fit the normalization parameters of the track features and the vertex targets into params/ (torch-free)."""
    from sv.scripts.dataset.compute_norm_par import main

    argv = ["--input", input_path]
    argv += _opt("--output", output)
    argv += _opt("--vertex-output", vertex_output)
    argv += _opt("--max-files", max_files)
    argv += _opt("--max-events", max_events)
    if dry_run:
        argv.append("--dry-run")
    sys.exit(main(argv))


# ------------------------------------------------------------------------------------------------ plot


@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(dir_okay=False),
              help="Prediction parquet written by the evaluation (<ckpt>__test[_<suff>].parquet).")
@click.option("--outdir", required=True, help="Directory for the plots and metrics.")
@click.option("--track-collection", default="track", show_default=True, help="Track collection of the file.")
@click.option("--vertex-collection", default="GV", show_default=True, help="Vertex collection of the file.")
@click.option("--max-events", type=int, default=None, help="Use only the first N events.")
def plot(input_path, outdir, track_collection, vertex_collection, max_events):
    """Run the SV plot chain on a prediction file (torch-free; awkward, numpy, pyarrow and matplotlib suffice)."""
    from sv.scripts.plot.run_sv_plot_chain import run_chain

    try:
        result = run_chain(input_path, outdir, track_collection=track_collection, vertex_collection=vertex_collection,
                           max_events=max_events)
    except Exception:  # noqa: BLE001 - the file could not be loaded
        traceback.print_exc()
        click.echo(f"[plot] could not load {input_path}")
        sys.exit(2)
    sys.exit(1 if result["failures"] else 0)


if __name__ == "__main__":
    cli(prog_name="maskformer-sv")
