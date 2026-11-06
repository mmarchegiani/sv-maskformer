# HGCAL Offline Reconstruction

Machine-learning reconstruction of HGCAL showers: MaskFormer models (built on [hepattn](https://github.com/mmarchegiani/hepattn))
that cluster the objects of an event (rechits, layer clusters or TICL tracksters) into particles and regress
their properties, trained and evaluated on CMS simulation. Everything runs through one command, `maskformer`.

The same MaskFormer machinery (`src/maskformer/`) also reconstructs **secondary vertices** from the tracks of an
event (`src/sv/`, command `maskformer-sv`, configs in `configs/sv/`), reproducing
[Secondary vertex reconstruction with MaskFormers](https://arxiv.org/abs/2312.12272) on CMS ttbar simulation:
see [configs/sv/README.md](configs/sv/README.md).

## Setup

The code runs inside a container image with every dependency installed:

```bash
source setup_env.sh        # opens a shell in the image; -v prints the image and commit dates
```

> [!WARNING]
> The image on `/cvmfs` is rebuilt at every commit on `master`, with some latency: a warning is printed when
> it is older than the checkout. `setup_env.sh` puts this checkout's `src/` first on `PYTHONPATH`, so the
> code you edit is the code that runs, including the `maskformer` command.

Outside the image, in a virtual environment: `pip install -e .` installs the package, its dependencies
(`pyproject.toml`) and the `maskformer` command. Without installing, run `PYTHONPATH=src python -m hgcal ...`
from the repository root.

## Datasets

```bash
bash fetch_datasets.sh [BASE_DIR]    # training taus and single-particle evaluation sets, from the CMU XRootD server
```

`BASE_DIR` defaults to `./data`; point it at a location with space (EOS, `/afs/.../work`). Then set the
`data:` paths of your config, see [configs/README.md](configs/README.md).

## Usage

```bash
maskformer --help                   # the commands; maskformer <command> --help for the options

# train: the config selects the model, the data and the normalization parameters
maskformer train --config configs/trackster_clue3dhigh/simtrackster.yaml

# evaluate a run: predictions, losses and the evaluation plots next to its best checkpoint
maskformer evaluate --config logs/<run>/config.yaml --test-path /path/to/test.parquet

# plot a prediction file again (no GPU or torch needed, e.g. in an LCG view)
maskformer plot --input logs/<run>/ckpts/<ckpt>__test_<suffix>.parquet --outdir plots/<run> --config logs/<run>/config.yaml
```

Any Lightning argument can be appended to `train` and `evaluate`, e.g. `--trainer.max_epochs 5` or `--data.num_workers 0`.

## Where to look next

| | |
|---|---|
| [src/README.md](src/README.md) | the source layout (`maskformer/` general, `hgcal/` HGCAL-specific), every command and option, checkpoints, evaluation outputs and plots, job submission on falcon and lxplus, energy-resolution tooling |
| [configs/README.md](configs/README.md) | the model configs, grouped by input type (`rechit/`, `layercluster/`, `trackster_clue3dhigh/`, `toycalo/`, `test/`), each folder with its own README |
| [params/README.md](params/README.md) | the normalization parameters of the inputs and targets, and how to refit them |
| [docs/](docs/) | design notes (normalization fix) and review findings |
