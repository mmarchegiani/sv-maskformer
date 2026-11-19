# Secondary Vertex Reconstruction with MaskFormers

Machine-learning reconstruction of **secondary vertices** (SV): MaskFormer models (built on [hepattn](https://github.com/mmarchegiani/hepattn))
that cluster the tracks of an event into heavy-flavour decay vertices and regress their properties, trained and
evaluated on CMS simulation. Everything runs through one command, `maskformer`.

This repository reproduces [Secondary vertex reconstruction with MaskFormers](https://arxiv.org/abs/2312.12272) on CMS ttbar simulation:
see [configs/sv/README.md](configs/sv/README.md).

## Setup

The code runs inside a container image with every dependency installed:

```bash
source setup_env.sh        # opens a shell in the image; -v prints the image and commit dates
```

`setup_env.sh` puts this checkout's `src/` first on `PYTHONPATH`, so the code you edit is the code that runs.

Outside the image, in a virtual environment: `pip install -e .` installs the package, its dependencies
(`pyproject.toml`) and the `maskformer` command. Without installing, run `PYTHONPATH=src python -m sv ...`
from the repository root.

## Usage

```bash
maskformer --help                # the commands; maskformer <command> --help for the options

# convert the ROOT ntuples to parquet, split into train/ val/ test/
maskformer convert '/path/to/ntuples/*.root' --outdir /path/to/parquet --split 0.8 0.1 0.1

# fit the normalization parameters of the track features and the vertex targets into params/
maskformer norm-par --input /path/to/parquet/train

# train: the config selects the model, the data and the normalization parameters
maskformer train --config configs/sv/gv_tracks.yaml

# evaluate a run: predictions, losses and the evaluation plots next to its best checkpoint
maskformer evaluate --config logs/<run>/config.yaml --test-path /path/to/parquet/test

# plot a prediction file again (no GPU or torch needed, e.g. in an LCG view)
maskformer plot --input logs/<run>/ckpts/<ckpt>__test_<suffix>.parquet --outdir plots/<run>
```

Any Lightning argument can be appended to `train` and `evaluate`, e.g. `--trainer.max_epochs 5` or `--data.num_workers 0`.

## Where to look next

| | |
|---|---|
| [configs/sv/README.md](configs/sv/README.md) | the problem and its mapping onto the MaskFormer, the model configs and their structure, the full workflow, the evaluation outputs |
| [src/README.md](src/README.md) | the source layout (`maskformer/` general, `sv/` SV-specific), every command and option, checkpoints, job submission on falcon, lxplus and HTCondor |
| [params/README.md](params/README.md) | the normalization parameters of the inputs and targets, and how to refit them |
