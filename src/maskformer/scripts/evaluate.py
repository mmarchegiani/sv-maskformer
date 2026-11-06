"""Evaluation entry point of a MaskFormer experiment, powered by the Lightning CLI (`test`).

`EvalCLI` runs the test loop (the run's config selects the model and the data; without `--ckpt_path` the best
checkpoint next to the config is used, see hepattn's CLI) and, once the prediction writer has written its file,
calls the `plot` hook. The experiment binds its classes and, optionally, a subclass with its plotting:

    from maskformer.scripts.evaluate import EvalCLI, main as evaluate_main
    class MyEvalCLI(EvalCLI):
        def plot(self, writer, parquet, sc): ...
    def main(args=None):
        evaluate_main(args, model_class=..., datamodule_class=..., cli_class=MyEvalCLI)

`--skip_plots true` skips the hook. A plotting failure never fails the evaluation.
"""

import traceback
from pathlib import Path

import comet_ml  # noqa: F401  (must be imported before torch/lightning)
from lightning.pytorch.cli import ArgsType

from maskformer.callbacks.prediction_writer import PredictionWriterBase
from maskformer.scripts.lightning_cli import MaskformerCLI


class EvalCLI(MaskformerCLI):
    """The training CLI plus a plotting hook after `test`."""

    def add_arguments_to_parser(self, parser) -> None:
        super().add_arguments_to_parser(parser)
        parser.add_argument(
            "--skip_plots",
            type=bool,
            default=False,
            help="Do not run the plotting after the test loop (`--skip_plots true`).",
        )

    # LightningCLI calls `after_<subcommand>` once the subcommand has returned
    def after_test(self) -> None:
        sc = self.config[self.subcommand]
        if sc.get("skip_plots"):
            print("[plots] --skip_plots set: not running the plotting chain")
            return
        if not self.trainer.is_global_zero:
            return
        writer = next((c for c in self.trainer.callbacks if isinstance(c, PredictionWriterBase)), None)
        if writer is None:
            print("[plots] no prediction writer among trainer.callbacks: nothing to plot")
            return
        if not writer.plot_chain_readable:
            print(f"[plots] {type(writer).__name__} writes a file the plot chain does not read: no plots")
            return
        parquet = Path(getattr(writer, "written_output_path", None) or writer.output_path)
        if not parquet.is_file():
            print(f"[plots] prediction file {parquet} not found: nothing to plot")
            return
        try:
            self.plot(writer, parquet, sc)
        except Exception:  # noqa: BLE001 - the prediction file is written, never fail here
            traceback.print_exc()
            print("[plots] plotting failed (see traceback above); the prediction file is unaffected")

    def plot(self, writer: PredictionWriterBase, parquet: Path, sc) -> None:
        """Plot the prediction file `parquet` written by `writer`; `sc` is the parsed `test` config."""
        print("[plots] no plotting configured for this experiment")


def main(args: ArgsType = None, *, model_class, datamodule_class, cli_class=EvalCLI, **cli_kwargs) -> None:
    cli_class(
        model_class=model_class,
        datamodule_class=datamodule_class,
        args=args,
        parser_kwargs={"default_env": True},
        **cli_kwargs,
    )
