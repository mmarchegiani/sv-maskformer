"""Evaluation entry point of the SV MaskFormer (`maskformer-sv evaluate`), powered by the Lightning CLI.

Loads the best (or a given) checkpoint, runs inference on a test parquet file, writes the predictions next to the
checkpoint (`sv.callbacks.prediction_writer.SVPredictionWriter`: the test file plus the prediction branches) and the
mean test losses, then runs the SV plot chain (src/sv/scripts/plot/run_sv_plot_chain.py) on that file.

    maskformer-sv evaluate --config logs/<run>/config.yaml --test-path /path/to/test.parquet

    python src/sv/scripts/evaluate.py test --config logs/<run>/config.yaml --data.test_path /path/to/test.parquet \\
        --trainer.precision 32-true --matmul_precision highest

Omit --ckpt_path to use the best checkpoint (lowest val loss) next to the config. The plots go to
`<ckpt_dir>/<ckpt_stem>__test[_<test_suff>]_plots/` (composition/, efficiency/, fakes/, classification/, regression/, tracks/,
metrics.json, summary.txt); `--plot_dir` moves them, `--skip_plots true` skips them, `--plot_max_events N` plots the
first N events only. The chain runs as a subprocess of the same interpreter (`SV_PLOT_PYTHON` overrides it); a
plotting failure never fails the evaluation, the command to rerun the chain by hand is printed instead.
"""

from maskformer.scripts.evaluate import EvalCLI, main as evaluate_main  # first: imports comet_ml before torch

import gc  # noqa: E402
import os  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

from lightning.pytorch.cli import ArgsType  # noqa: E402

from maskformer.models.wrapper import MaskformerWrapper  # noqa: E402

from sv.dataset.dataset_sv import SVDataModule  # noqa: E402

PLOT_CHAIN = Path(__file__).resolve().parent / "plot" / "run_sv_plot_chain.py"


class SVEvalCLI(EvalCLI):
    """The evaluation CLI plus the SV plot chain after `test`."""

    def add_arguments_to_parser(self, parser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument("--plot_dir", type=Optional[str], default=None,
                            help="Directory for the evaluation plots. Default: <ckpt_dir>/<ckpt_stem>__test[_<test_suff>]_plots/.")
        parser.add_argument("--plot_max_events", type=Optional[int], default=None,
                            help="Plot only the first N events of the prediction file (smoke test).")

    def plot(self, writer, parquet: Path, sc) -> None:
        plot_dir = Path(sc["plot_dir"]) if sc.get("plot_dir") else parquet.with_name(parquet.stem + "_plots")
        track_collection = writer.track_collection
        vertex_collection = writer.vertex_collection
        # release the test dataset before spawning the chain
        writer.ds = None
        datamodule = self.trainer.datamodule
        if datamodule is not None and hasattr(datamodule, "test_dset"):
            datamodule.test_dset = None
        gc.collect()

        cmd = [os.environ.get("SV_PLOT_PYTHON", sys.executable), str(PLOT_CHAIN), "--input", str(parquet), "--outdir", str(plot_dir),
               "--track-collection", track_collection, "--vertex-collection", vertex_collection]
        if sc.get("plot_max_events") is not None:
            cmd += ["--max-events", str(sc["plot_max_events"])]
        print("-" * 100)
        print("[plots] running the SV plot chain:\n  " + shlex.join(cmd), flush=True)
        rc = subprocess.run(cmd, check=False).returncode  # noqa: S603
        if rc == 0:
            print(f"[plots] done -> {plot_dir}")
        else:
            by_cli = ["maskformer-sv", "plot", "--input", str(parquet), "--outdir", str(plot_dir)]
            print(f"[plots] plot chain exited with code {rc}; rerun it by hand with\n  {shlex.join(cmd)}\n"
                  f"or, in an environment with the package, with\n  {shlex.join(by_cli)}")
        print("-" * 100)


def main(args: ArgsType = None) -> None:
    evaluate_main(args, model_class=MaskformerWrapper, datamodule_class=SVDataModule, cli_class=SVEvalCLI)


if __name__ == "__main__":
    main()
