"""Training entry point of a MaskFormer experiment, powered by the Lightning CLI.

The experiment binds its model wrapper and data module:

    from maskformer.scripts.train import main as train_main
    def main(args=None):
        train_main(args, model_class=MaskformerWrapper, datamodule_class=MyDataModule)

The config alone then defines what is trained (`model:`, `data:`, `trainer:`); there is no default config.
"""

import comet_ml  # noqa: F401  (must be imported before torch/lightning)
import torch._inductor.config
torch._inductor.config.shape_padding = False  # Prevents OOM during compile-time pad_mm benchmarking

from lightning.pytorch.cli import ArgsType

from maskformer.scripts.lightning_cli import MaskformerCLI


def main(args: ArgsType = None, *, model_class, datamodule_class, cli_class=MaskformerCLI, **cli_kwargs) -> None:
    """Run the Lightning CLI (`fit`, `validate`, ...; `args` holds the subcommand) with the experiment's classes.
    `cli_kwargs` go to `cli_class` (e.g. the legacy class-path map of `MaskformerCLI`)."""
    cli_class(
        model_class=model_class,
        datamodule_class=datamodule_class,
        args=args,
        parser_kwargs={"default_env": True},
        **cli_kwargs,
    )
