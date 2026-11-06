"""Training entry point of the SV MaskFormer (`maskformer-sv train`), powered by the Lightning CLI.

The config alone defines what is trained: the model under `model:` (hepattn's MaskFormer with the classification, mask
and regression tasks), the callbacks, and under `data:` the files, the input features and targets, the two scale
dicts and the track selection of `SVDataModule`. There is no default config: pass `--config`.

    maskformer-sv train --config configs/sv/gv_tracks.yaml [Lightning arguments]
    python src/sv/scripts/train.py fit --config ...          # the same without the click wrapper
"""

from maskformer.scripts.train import main as train_main  # first: imports comet_ml before torch/lightning
from lightning.pytorch.cli import ArgsType  # noqa: E402

from maskformer.models.wrapper import MaskformerWrapper  # noqa: E402

from sv.dataset.dataset_sv import SVDataModule  # noqa: E402


def main(args: ArgsType = None) -> None:
    train_main(args, model_class=MaskformerWrapper, datamodule_class=SVDataModule)


if __name__ == "__main__":
    main()
