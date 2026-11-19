# The `maskformer` command and the source layout

## Package layout

`src/` holds two packages:

| package | contents | may import |
|---|---|---|
| `maskformer/` | the **general** MaskFormer code: the Lightning wrapper of hepattn's model (`models/wrapper.py`), the task overrides and the sectored decoder (`models/`), the checkpoint, early-stopping and prediction-writer base callbacks (`callbacks/`), parquet/path/finiteness utilities (`utils/`), and the generic entry points: the Lightning-CLI wrappers for training and evaluation, the click building blocks and the config migration (`scripts/`) | hepattn, torch, lightning |
| `sv/` | the **secondary-vertex** experiment (tracks -> heavy-flavour decay vertices, after arXiv:2312.12272, on the BTV ntuples): collection knowledge and feature derivations (`dataset/collections.py`, `dataset/features.py`, torch-free), `SVDataset`/`SVDataModule` (`dataset/dataset_sv.py`), `SVPredictionWriter` (`callbacks/`), the `maskformer` command line with the ROOT -> parquet converter, the normalization fit, the training/evaluation bindings and the plot chain (`scripts/`). See [configs/sv/README.md](../configs/sv/README.md) | `maskformer` |

The rule, enforced by `tests/test_import_rule.py`: an experiment package (`sv`) imports `maskformer`, and
`maskformer` never imports an experiment. A new experiment reuses `maskformer` as is and provides its own dataset,
writers and bindings next to `sv`. Class names follow the split: the general classes carry no experiment prefix
(`MaskformerWrapper`, `MaskformerCLI`, `RowwiseObjectHitMaskTask`, `ConfigurableIncidenceBasedRegressionTask`,
`ConfigurableObjectRegressionTask`, `PredictionWriterBase`), the experiment ones keep theirs (`SVDataset`,
`SVDataModule`, `SVPredictionWriter`).

## The commands

Every task runs through `maskformer`, a [click](https://click.palletsprojects.com) command
(`src/sv/scripts/cli.py`, built from the pieces of `src/maskformer/scripts/cli.py`):

| command | wraps | purpose |
|---|---|---|
| `maskformer convert` | `sv/scripts/dataset/convert_root.py` | convert the BTV ROOT ntuples to the parquet layout of the SV dataset (torch-free) |
| `maskformer norm-par` | `sv/scripts/dataset/compute_norm_par.py` | fit the normalization parameters into `params/` ([params/README.md](../params/README.md)) (torch-free) |
| `maskformer train` | `sv/scripts/train.py` -> `maskformer/scripts/train.py` | train a MaskFormer through the Lightning CLI |
| `maskformer evaluate` | `sv/scripts/evaluate.py` -> `maskformer/scripts/evaluate.py` | run a trained model on a test file: predictions, losses, the plot chain |
| `maskformer plot` | `sv/scripts/plot/run_sv_plot_chain.py` | the plot chain on a prediction file (torch-free) |

`maskformer --help` lists the commands, `maskformer <command> --help` the options. The workflow, the configs
and the evaluation outputs are documented in [configs/sv/README.md](../configs/sv/README.md). The entry points also
run on their own (`python src/sv/scripts/train.py fit --config ...`,
`python src/sv/scripts/plot/run_sv_plot_chain.py --input ...`), with `src/` on `PYTHONPATH`.

## Environments

- **Container** (`source setup_env.sh`): everything, GPU included. `setup_env.sh` exports
  `PYTHONPATH=<checkout>/src` into the shell, so the checkout's code is what runs: `python -m sv ...`, the form
  every job script uses.
- **LCG view** (`source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh`): no torch, but
  enough for `convert`, `norm-par`, `plot` and the torch-free tests. Run
  `PYTHONPATH=src:$PYTHONPATH python -m sv ...` (keep the view's own `PYTHONPATH`).
- **A virtual environment**: `pip install -e .` installs the dependencies of `pyproject.toml` and the command.

## Lightning arguments (`train`, `evaluate`)

`train` and `evaluate` front hepattn's `LightningCLI` (`fit` and `test`). Their options cover the usual knobs;
**any other argument is forwarded unchanged** to the Lightning CLI, after the options and in the order given, so a
forwarded argument overrides an option (`--debug --data.num_workers 4` ends with 4 workers). Examples:
`--trainer.max_epochs 5`, `--data.num_workers=0`, `--trainer.callbacks+=...`,
`--model.model.init_args.encoder.init_args.attn_type torch`, `--print_config`. Do not pass `fit`/`test`
yourself: the commands add them. `--help` prints the command's own options, `--lightning-help` the full
Lightning help of the sub-command.

Every `scale_dict_path` of the parsed config is resolved before anything is instantiated
(`maskformer/scripts/lightning_cli.py`, `maskformer.utils.paths`): `params/<name>.yaml` is found from any
directory. The resolved path is what the run's `config.yaml` records.

## `maskformer train`

```bash
maskformer train --config configs/sv/gv_tracks.yaml
maskformer train -c configs/sv/gv_tracks_small.yaml --debug --max-epochs 1 --trainer.limit_train_batches 5   # a quick pass
```

| option | Lightning argument | |
|---|---|---|
| `-c, --config PATH` (required, repeatable) | `--config` | later configs override earlier ones |
| `--name NAME` | `--name` | the run folder is `logs/<name>_<timestamp>` (default: the config's `name`) |
| `--ckpt-path PATH` | `--ckpt_path` | resume from a checkpoint |
| `--num-workers N` | `--data.num_workers` | |
| `--max-epochs N` | `--trainer.max_epochs` | |
| `--devices`, `--precision` | `--trainer.devices`, `--trainer.precision` | |
| `--matmul-precision` | `--matmul_precision` | `highest`, `high`, `medium`, `low` |
| `--debug` | `--data.num_workers 0 --trainer.num_sanity_val_steps 0` | single worker, no validation sanity check |

There is no default config: the config alone defines the model, the data and the normalization parameters
([configs/sv/README.md](../configs/sv/README.md)).

### Checkpoints and early stopping

Checkpoints go to `logs/<run>/ckpts/`. hepattn's `Checkpoint` callback writes every epoch as
`epoch=NNN-val_loss=X.XXXXX.ckpt`; the `BestCheckpoint` callback (`src/maskformer/callbacks/best_checkpoint.py`) keeps, next
to them, a single copy of the best epoch so far as `best-epoch=NNN-val_loss=X.XXXXX.ckpt`, replacing the previous
`best-*` file whenever a later epoch lowers `val/loss`.

The `EarlyStopping` callback (`src/maskformer/callbacks/early_stopping.py`, a thin layer over Lightning's) ends the training
before `trainer.max_epochs` when the total validation loss `val/loss` has not decreased for `patience` consecutive
epochs (a NaN/inf `val/loss` also stops it).

### Non-finite (NaN/inf) guards

A NaN in a single event turns the epoch-mean `val/loss` into NaN while the on-step `train/loss` (logged every
`log_every_n_steps` batches) can still look fine, and an L1 regression term with a NaN target even keeps finite
gradients, so such a run trains on silently. The pipeline therefore asserts finiteness where a bad value can enter
(`src/maskformer/utils/finite_checks.py`):

- `SVDataset`: the scale dicts, the raw branches, the domain of the `log`/`log1p` transforms, and the scaled track
  features and vertex targets.
- `MaskformerWrapper`: every batch on arrival and every loss term after `model.loss`. The first non-finite loss
  raises an `AssertionError` naming the terms and the event numbers, after saving the batch (`inputs`, `targets`,
  `losses`) to `<log_dir>/nonfinite_<stage>_step<N>_batch<M>.pt` (or `--model.nonfinite_dump_dir`) so it can be
  replayed on a CPU.

`tests/test_finite_checks.py` exercises the checks on synthetic tensors.

## `maskformer evaluate`

```bash
maskformer evaluate --config logs/<run>/config.yaml --test-path /path/to/test.parquet
maskformer evaluate -c logs/<run>/config.yaml --test-path <file> --plot-max-events 50          # smoke test of the plots
```

| option | Lightning argument | default |
|---|---|---|
| `-c, --config PATH` (required, one) | `--config` | the run's `config.yaml`; without `--ckpt-path` the best checkpoint (lowest `val_loss`) in `ckpts/` next to it is used |
| `--ckpt-path PATH` | `--ckpt_path` | |
| `--test-path PATH` | `--data.test_path` | the config's test split; a parquet file or a directory of them |
| `--test-suff SUFF` | `--data.test_suff` | the stem of `--test-path`; names the outputs `<ckpt>__test_<suff>.*` |
| `--num-workers N` | `--data.num_workers` | 0 |
| `--precision` | `--trainer.precision` | `32-true` |
| `--matmul-precision` | `--matmul_precision` | `highest` |
| `--root-dir DIR` | `--trainer.default_root_dir` | where the outputs of the run go |
| `--attn-type {flash-varlen,torch}` | `--model.model.init_args.encoder.init_args.attn_type` | `torch` on GPUs without flash attention (lxplus T4s) |
| `--plot-dir DIR` | `--plot_dir` | `<ckpt_dir>/<ckpt>__test[_<suff>]_plots/` |
| `--skip-plots` | `--skip_plots true` | |
| `--plot-max-events N` | `--plot_max_events` | plot only the first N events |

Outputs:

- a parquet file next to the checkpoint, `<ckpt_name>__test[_<suff>].parquet`, with the test events plus the
  prediction branches (the module doc of `src/sv/callbacks/prediction_writer.py` lists them), and
  `<ckpt_name>__test[_<suff>]_losses.json` with the mean test loss per task;
- the plots of the chain, in `<ckpt_name>__test[_<suff>]_plots/` (or `--plot-dir`), described in
  [configs/sv/README.md](../configs/sv/README.md#outputs-of-the-evaluation). The chain runs as a subprocess of the
  same interpreter (`SV_PLOT_PYTHON` selects another one); if it fails the evaluation still succeeds and the
  commands to rerun it by hand (the script and `maskformer plot`) are printed.

## `maskformer plot`

The chain by hand, on any prediction file. It needs only awkward, numpy, pyarrow and matplotlib, so it runs in an
LCG view:

```bash
maskformer plot --input <ckpt_dir>/<ckpt>__test_<suff>.parquet --outdir plots/<run>
```

Options: `--track-collection` (default `track`), `--vertex-collection` (default `GV`), `--max-events N` for a
subset. Exit code 0 when every section ran, 1 when some failed, 2 when the file could not be loaded.

## Jobs (`jobs/`)

The job scripts run `python -m sv ...` inside the CVMFS image with `PYTHONPATH=<checkout>/src`, from the
checkout root.

| script | site | usage |
|---|---|---|
| `jobs/train_sv_job.sh` | falcon (Slurm) | `sbatch jobs/train_sv_job.sh [data_dir] [args]`: trains `configs/sv/gv_tracks.yaml` on the train/ val/ test/ folders of a converted SV dataset |
| `jobs/train_sv_lxplus.sh` | lxplus GPU | `bash jobs/train_sv_lxplus.sh [config] [args]`: the same on lxplus-gpu with the EOS files of the config (Kerberos-bound) |
| `jobs/train_sv_condor.sub` + `jobs/train_sv_condor.sh` | CERN GPU batch (HTCondor) | `condor_submit jobs/train_sv_condor.sub [smoke=1 flavour=longlunch] [name=... extra="..."]`: unattended training on an A100/H100 (Kerberos delegated by condor), evaluation of the best checkpoint on the test split, plots published to `/eos/user/m/mmarcheg/www/plots/ml4reco/sv-maskformers/<run>/evaluation/test/`; run outputs in `/eos/user/m/mmarcheg/BTV/sv_reconstruction/runs/job_<cluster>/`, STATUS file there; `run_dir=<run>` evaluates a finished run only |

## Developer notes

- Both packages are found by `pyproject.toml` (`packages.find` under `src/`, `maskformer*` and `sv*`); the console
  script is `maskformer = sv.scripts.cli:cli`.
- `maskformer/scripts/lightning_cli.py` holds `MaskformerCLI`, hepattn's CLI plus the scale-dict path resolution;
  `maskformer/scripts/train.py` and `evaluate.py` take the experiment's `model_class` / `datamodule_class` (and a
  `cli_class`), `sv/scripts/train.py` and `evaluate.py` bind `MaskformerWrapper`, `SVDataModule` and `SVEvalCLI`
  (the plot chain). The SV click group adds its own commands to the generic `train`/`evaluate` pieces of
  `maskformer/scripts/cli.py`.
- The prediction writers derive from `maskformer.callbacks.PredictionWriterBase` (output paths, un-scaling, loss
  summary, the parquet write) and implement `regression_transforms`, `reset_buffers`, `collect_batch` and
  `build_output`.
- The plot modules insert their own folder into `sys.path` and import their siblings as top-level modules
  (`from sv_eval_common import ...`), so that `run_sv_plot_chain.py` also runs by file path under another
  interpreter. Never import them package-qualified (`sv.scripts.plot.sv_eval_common`): there would be two module
  objects and `isinstance` checks between them would fail. `run_sv_plot_chain` (imported by the CLI) is the one
  exception by design.
- Tests: `tests/test_paths.py`, `test_configs.py`, `test_import_rule.py`, `test_sv_features.py`, `test_sv_cli.py`,
  `test_sv_convert.py` (a synthetic ntuple written with uproot) and `test_sv_plot_chain.py` run without torch
  (`PYTHONPATH=src:$PYTHONPATH python -m pytest tests/test_sv_cli.py ...` in an LCG view); the rest, including
  `test_sv_dataset.py` and `test_regression_task.py` (the SV regression task's loss and cost), needs the container
  (`PYTHONPATH=src python -m pytest tests`).
