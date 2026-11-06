# The `maskformer` command and the source layout

## Package layout

`src/` holds two packages:

| package | contents | may import |
|---|---|---|
| `maskformer/` | the **general** MaskFormer code: the Lightning wrapper of hepattn's model (`models/wrapper.py`), the task overrides and the sectored decoder (`models/`), the checkpoint, early-stopping and prediction-writer base callbacks (`callbacks/`), parquet/path/finiteness utilities (`utils/`), and the generic entry points: the Lightning-CLI wrappers for training and evaluation, the click building blocks and the config migration (`scripts/`) | hepattn, torch, lightning |
| `hgcal/` | everything **HGCAL-specific**: the dataset and its collection knowledge (`dataset/`), the prediction writers that write the HGCAL collections back into the input file (`callbacks/`), the `maskformer` command line, the training/evaluation bindings, the plot chain, the normalization fits and the toy-calorimeter tools (`scripts/`) | `maskformer` |
| `sv/` | the **secondary-vertex** experiment (tracks -> heavy-flavour decay vertices, after arXiv:2312.12272, on the BTV ntuples): collection knowledge and feature derivations (`dataset/collections.py`, `dataset/features.py`, torch-free), `SVDataset`/`SVDataModule` (`dataset/dataset_sv.py`), `SVPredictionWriter` (`callbacks/`), the `maskformer-sv` command line with the ROOT -> parquet converter, the normalization fit, the training/evaluation bindings and the plot chain (`scripts/`). See [configs/sv/README.md](../configs/sv/README.md) | `maskformer` |

The rule, enforced by `tests/test_import_rule.py`: the experiment packages (`hgcal`, `sv`) import `maskformer`,
`maskformer` never imports an experiment, and the experiments never import each other. A new experiment reuses
`maskformer` as is and provides its own dataset, writers and bindings next to `hgcal` and `sv`. Class names follow
the split: the general classes carry no experiment prefix (`MaskformerWrapper`, `MaskformerCLI`,
`RowwiseObjectHitMaskTask`, `ConfigurableIncidenceBasedRegressionTask`, `ConfigurableObjectRegressionTask`,
`PredictionWriterBase`), the experiment ones keep theirs (`HGCDataset`, `HGCDataModule`, `HGCPredictionWriter`,
`SectoredPredictionWriter`; `SVDataset`, `SVDataModule`, `SVPredictionWriter`).

## The commands

Every HGCAL task runs through `maskformer`, a [click](https://click.palletsprojects.com) command (`src/hgcal/scripts/cli.py`,
built from the pieces of `src/maskformer/scripts/cli.py`); the SV experiment has its own, `maskformer-sv`
(`src/sv/scripts/cli.py`, `python -m sv`): `train`, `evaluate`, `convert` (ROOT ntuples -> parquet), `norm-par` and
`plot`, documented in [configs/sv/README.md](../configs/sv/README.md). The `maskformer` commands:

| command | wraps | purpose |
|---|---|---|
| `maskformer train` | `hgcal/scripts/train.py` -> `maskformer/scripts/train.py` | train a MaskFormer through the Lightning CLI |
| `maskformer evaluate` | `hgcal/scripts/evaluate.py` -> `maskformer/scripts/evaluate.py` | run a trained model on a test file: predictions, losses, the plot chain |
| `maskformer plot` | `hgcal/scripts/plot/run_plot_chain.py` | the plot chain on a prediction file (torch-free) |
| `maskformer plot-toycalo` | `hgcal/scripts/plot/plot_toycalo_*.py` | toy-calorimeter exploration plots |
| `maskformer norm-par` | `hgcal/scripts/dataset/compute_norm_par.py` | fit the normalization parameters into `params/` ([params/README.md](../params/README.md)) |
| `maskformer verify-norm` | `hgcal/scripts/dataset/verify_norm.py` | check a config's normalization through `HGCDataset` |
| `maskformer migrate-config` | `maskformer/scripts/migrate_config.py` | rewrite run configs saved before the maskformer/hgcal split |

`maskformer --help` lists the commands, `maskformer <command> --help` the options. The entry points also run on
their own (`python src/hgcal/scripts/train.py fit --config ...`, `python src/hgcal/scripts/plot/run_plot_chain.py --input ...`),
with `src/` on `PYTHONPATH`.

## Environments

- **Container** (`source setup_env.sh`): everything, GPU included. The image installs the package, so `maskformer`
  is on `PATH` once the image has been rebuilt after the commit (an image built before the split still has the
  old `maskformers` script: use `python -m hgcal` until then); `setup_env.sh` exports `PYTHONPATH=<checkout>/src`
  into the shell, so the checkout's code (and CLI) is what runs. Until the image is rebuilt, and in every job
  script, the equivalent form is `python -m hgcal ...` with `PYTHONPATH=<checkout>/src`.
- **LCG view** (`source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh`): no torch, but
  enough for `maskformer plot`, `plot-toycalo` and the torch-free tests. Run
  `PYTHONPATH=src:$PYTHONPATH python -m hgcal ...` (keep the view's own `PYTHONPATH`).
- **A virtual environment**: `pip install -e .` installs the dependencies of `pyproject.toml` and the command.

## Lightning arguments (`train`, `evaluate`)

`train` and `evaluate` front hepattn's `LightningCLI` (`fit` and `test`). Their options cover the usual knobs;
**any other argument is forwarded unchanged** to the Lightning CLI, after the options and in the order given, so a
forwarded argument overrides an option (`--debug --data.num_workers 4` ends with 4 workers). Examples:
`--trainer.max_epochs 5`, `--data.num_workers=0`, `--trainer.callbacks+=...`,
`--model.model.init_args.encoder.init_args.attn_type torch`, `--print_config`. Do not pass `fit`/`test`
yourself: the commands add them. `--help` prints the command's own options, `--lightning-help` the full
Lightning help of the sub-command.

Every `scale_dict_path` of the parsed config (the data module's two dicts and the regression task's) is resolved
before anything is instantiated (`maskformer/scripts/lightning_cli.py`, `maskformer.utils.paths`): `params/<name>.yaml` is found from any
directory, and the `configs/<name>.yaml` of runs from before November 2026 is mapped onto `params/` with one
`[paths] ... using params/...` line in the log. The resolved path is what the run's `config.yaml` records.

## `maskformer train`

```bash
maskformer train --config configs/trackster_clue3dhigh/simtrackster.yaml
maskformer train -c configs/rechit/mergedsimcluster.yaml --debug --max-epochs 1        # a quick pass
maskformer train -c configs/trackster_clue3dhigh/simtrackster.yaml --data.data_dir <dir> --data.split '[0.8,0.1,0.1]'
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
([configs/README.md](../../configs/README.md)). A sectored-decoder model ([configs/toycalo/](../../configs/toycalo/README.md))
is selected by its config (`data.n_sectors` and the sectored class paths); the command is the same.

### Checkpoints and early stopping

Checkpoints go to `logs/<run>/ckpts/`. hepattn's `Checkpoint` callback writes every epoch as
`epoch=NNN-val_loss=X.XXXXX.ckpt`; the `BestCheckpoint` callback (`src/maskformer/callbacks/best_checkpoint.py`) keeps, next
to them, a single copy of the best epoch so far as `best-epoch=NNN-val_loss=X.XXXXX.ckpt`, replacing the previous
`best-*` file whenever a later epoch lowers `val/loss`.

The `EarlyStopping` callback (`src/maskformer/callbacks/early_stopping.py`, a thin layer over Lightning's) ends the training
before `trainer.max_epochs` when the total validation loss `val/loss` has not decreased for `patience` consecutive
epochs (a NaN/inf `val/loss` also stops it). Both are enabled in every training config; remove the `EarlyStopping`
entry to always train for `max_epochs` (the YAML is in [configs/README.md](../../configs/README.md#checkpoints-and-early-stopping)).

### Non-finite (NaN/inf) guards

A NaN in a single event turns the epoch-mean `val/loss` into NaN while the on-step `train/loss` (logged every
`log_every_n_steps` batches) can still look fine, and an L1 regression term with a NaN target even keeps finite
gradients, so such a run trains on silently. The pipeline therefore asserts finiteness where a bad value can enter
(`src/maskformer/utils/finite_checks.py`):

- `HGCDataset`: the scale dicts (finite shift, non-zero scale), every raw branch of the pair once per file, the
  domain of the `log`/`log1p` energy transforms, the energy shares of the TICL association table (finite, in
  [0, 1], summing to at most 1 per trackster), and per event the scaled node inputs, the scaled particle targets
  and the incidence matrix (finite, in [0, 1], node columns or particle rows summing to at most 1).
- `MaskformerWrapper`: every batch on arrival (valid nodes, valid particles, the dense incidence target) and every
  loss term after `model.loss`. The first non-finite loss raises an `AssertionError` naming the terms and the
  event numbers, after saving the batch (`inputs`, `targets`, `losses`) to
  `<log_dir>/nonfinite_<stage>_step<N>_batch<M>.pt` (or `--model.nonfinite_dump_dir`) so it can be replayed on a CPU.

`tests/test_finite_checks.py` exercises the checks on synthetic tensors.

## `maskformer evaluate`

```bash
maskformer evaluate --config logs/<run>/config.yaml --test-path /path/to/test.parquet
maskformer evaluate -c logs/<run>/config.yaml --test-path <file> --plot-max-events 50          # smoke test of the plots
maskformer evaluate -c <model_dir>/config.yaml --root-dir <model_dir> --test-path <file> --attn-type torch \
    --data.data_dir null --data.split null                                                      # a run copied from another cluster
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
| `--plot-max-events N` | `--plot_max_events` | plot only the first N events (the file is still read in full) |
| `--plot-n-display N` | `--plot_n_display` | event displays (5; 0 disables them) |

The defaults reproduce what the job scripts always passed (full precision, a single worker). Outputs:

- a parquet file next to the checkpoint, `<ckpt_name>__test[_<suff>].parquet`, with the same schema as the input
  test file plus the prediction branches (the `HGCPredictionWriter` docstring in `src/hgcal/callbacks/prediction_writer.py`
  lists them), and `<ckpt_name>__test[_<suff>]_losses.json` with the mean test loss per task;
- the plots of the chain below, in `<ckpt_name>__test[_<suff>]_plots/` (or `--plot-dir`). The chain runs as a
  subprocess of the same interpreter (`HGCAL_PLOT_PYTHON` selects another one, e.g. one with `mplhep`); if it fails
  the evaluation still succeeds and the commands to rerun it by hand (the script and `maskformer plot`) are printed.
- a sectored-decoder run (`SectoredPredictionWriter`) gets its predictions and losses, and the plots are skipped
  with a note: the chain reads the per-particle schema only.

### Evaluation plots

- `resolution/`, `resolution_by_class/`, `efficiency/`, `diagnostics/`: per-cluster resolutions and clustering
  efficiency, plus `metrics.json`/`summary.txt`;
- `cluster_properties/`: true vs predicted distribution of every regressed feature (e, eta, cosphi, sinphi, r, and
  the boundary targets when regressed), `response/` the 1D response `pred/true` per feature and `response/by_<feat>/`
  the response of every feature in bins of every (true) feature;
- `boundary_position/`: the entrance-position performance when `boundary_x`/`boundary_y` were regressed
  ([configs/trackster_clue3dhigh/README.md](../../configs/trackster_clue3dhigh/README.md#entrance-position-of-the-particle-boundary-targets)):
  predicted vs true x and y, the 2D residual with its d68 circle, the transverse distance
  `d = |(x,y)_pred - (x,y)_true|` against two clustering-only baselines (the energy-weighted barycenter of the nodes
  the model assigned to the particle, and of the truth-assigned nodes, projected onto the boundary z along the line
  from the origin), `d` vs true energy / eta / class, the x and y residuals vs the true x and y (edge effects) and
  the entry-point maps; `metrics.json` gets a `boundary_position` block and `summary.txt` an "entrance position"
  section; `event_reconstruction/weighted_boundary_{x,y}_*` give the energy-weighted entry point of the whole event;
- `topocluster_features/raw/` and `topocluster_features/transform/`: the node input features (`data.inputs.topocluster`)
  as read from the file and after the scale-dict transform the dataset applies (`data.scale_dict_path`);
- `event_reconstruction/`: the event as one object, i.e. the sum of all predicted clusters (energy sum,
  energy-weighted eta/cosphi/sinphi/r) against the true particles, what matters for a photon reconstructed as
  e+e-. Only matched queries enter (the kinematics of fake queries are not stored; their count is plotted);
- `event_display/`: 3D views and 2D projections (y-x, x-z, y-z, rho-z) of the first events, true and predicted
  clusters as rings sized by their energy, nodes as dots;
- `energy/`: the energy deep-dive with `energy_metrics.json`/`energy_summary.txt`.

## `maskformer migrate-config`

```bash
maskformer migrate-config logs/<run>/config.yaml            # rewrites in place, keeps config.yaml.bak
maskformer migrate-config --dry-run <model_dir>/config.yaml  # only reports the changes
```

A run config saved before the maskformer/hgcal split (November 2026) names the class paths of the old layout
(`callbacks.HGCPredictionWriter`, `callbacks.best_checkpoint.BestCheckpoint`, `models.tasks.*`, ...) and, if older
still, `configs/hgcal_*` scale dicts. `train` and `evaluate` refuse such a config with a message naming this command
(a stale installed copy of the old packages could otherwise be imported silently); `migrate-config` rewrites the
file line by line (class paths through the map in `hgcal/scripts/legacy.py`, `configs/<name>` -> `params/<name>`),
keeps the original as `<file>.bak`, re-parses the result and is idempotent. Run it once per model folder on EOS or
XRootD before evaluating runs from before the split.

## `maskformer plot`

The chain by hand, on any prediction file. It needs only awkward, numpy, pyarrow and matplotlib (`mplhep` is
optional), so it runs in an LCG view:

```bash
maskformer plot --input <ckpt_dir>/<ckpt>__test_<suff>.parquet --outdir plots/<run> --config logs/<run>/config.yaml
maskformer plot --input <file> --outdir <dir> --scale-dict params/hgcal_trackster_clue3dhigh_var_transform.yaml --node-features e,eta,cosphi,sinphi,r
```

`--config <run>/config.yaml` takes from the run's `data:` block the collections, the association table and
`min_link_fraction` of the trackster pair, the node input features and the node scale dict, which is what the
evaluation passes to the chain itself. The other options override it: `--topocluster-collection` /
`--particle-collection` (default: auto-detected from the file, the collections carrying `*_pred_match_idx` and
`*_pred_valid`), `--association-collection` and `--min-link-fraction` (trackster pair), `--scale-dict` (the node
dict of the training config; without it only `topocluster_features/raw/` is written) and `--node-features`
(comma-separated or repeated; default `e eta cosphi sinphi r`); `--max-events N` for a subset, `--skip-energy`,
`--n-display N` / `--skip-display` for the event displays. Exit code 0 when every section ran, 1 when some failed
(the tracebacks are in `<outdir>/plot_chain_failures.txt`), 2 when the file could not be loaded.

The plot modules (`plot_maskformer_eval.py`, `energy_investigation.py`, `cluster_properties.py`, `boundary_position.py`,
`topocluster_features.py`, `event_reconstruction.py`, `event_display.py`) can also be run on their own with
`--input`/`--outdir`; `hgcal_eval_common.py` holds the shared loader (`EvalData`), the pair detection and the style.

## `maskformer plot-toycalo`

```bash
maskformer plot-toycalo --input data/photons_100GeV_eta20.parquet --outdir plots/toycalo [--max-events N] [--n-display 3] [--only rechits ...]
```

runs the four toy-calorimeter exploration scripts (`plot_toycalo_rechits.py`, `plot_toycalo_hit_particle.py`,
`plot_toycalo_particles.py`, `plot_toycalo_gun.py`) into `<outdir>/{rechits,hit_particle,particles,gun_parameters}/`;
`--only` selects sections.

## `maskformer norm-par` and `maskformer verify-norm`

Fitting the normalization parameters and checking a config's normalization end to end: see
[params/README.md](../../params/README.md).

## Jobs (`jobs/`)

The job scripts run `python -m hgcal ...` inside the CVMFS image with `PYTHONPATH=<checkout>/src`, from the
checkout root.

| script | site | usage |
|---|---|---|
| `jobs/gpu_job.sh` | falcon (Slurm) | `sbatch jobs/gpu_job.sh`: trains `configs/rechit/mergedsimcluster.yaml` |
| `jobs/train_trackster_job.sh` | falcon | `sbatch jobs/train_trackster_job.sh [data_dir]`: trains `configs/trackster_clue3dhigh/simtrackster.yaml` on a directory of parquet files with a `[0.8, 0.1, 0.1]` split |
| `jobs/eval_job.sh` | falcon | `sbatch jobs/eval_job.sh <run_dir> <test.parquet or file_list.txt> [extra args]`: evaluates every listed file and moves the outputs to `<run_dir>/eval/<stem>/` |
| `jobs/eval_lxplus.sh` | lxplus GPU | `bash jobs/eval_lxplus.sh <model_dir> [file_list.txt] [extra args]`: Kerberos-bound evaluation of EOS files, plots published to the EOS www area with a web index |
| `jobs/norm_par_job.sh` | falcon | `TOPO=... PART=... DATA_DIR=... sbatch jobs/norm_par_job.sh`: fits two scale dicts into `params/` |
| `jobs/copy_files.sh`, `jobs/photon_eval_files.txt` | | copies the photon evaluation sets from XRootD to EOS; the file list the lxplus evaluation defaults to |
| `jobs/train_sv_job.sh` | falcon | `sbatch jobs/train_sv_job.sh [data_dir] [args]`: trains `configs/sv/gv_tracks.yaml` on the train/ val/ test/ folders of a converted SV dataset |
| `jobs/train_sv_lxplus.sh` | lxplus GPU | `bash jobs/train_sv_lxplus.sh [config] [args]`: the same on lxplus-gpu with the EOS files of the config (Kerberos-bound) |

### Energy-resolution tooling (single-particle scans)

`jobs/resolution/` evaluates a trained model on mono-energetic single-particle sets (photons / pions / K0L at
2-300 GeV, one job per particle and energy) and produces resolution tables and plots. Everything runs inside the
CVMFS image through sbatch.

```bash
# one job per set; data in $EVAL_SETS/<particle>_E<E>/*.parquet, model in $RUN_DIR (config.yaml + ckpts/)
RUN_DIR=models/<run> EVAL_SETS=/path/to/eval_sets bash jobs/resolution/submit_all_cmu_sets.sh
# Gaussian fits of E_pred, sigma68 of E_pred/E_true and of the plain CLUE3D trackster sum (sumTS), per-set histograms
sbatch jobs/resolution/analysis_job.sh eval_gaussfit.py models/<run>/eval
# merged table (events, tracksters, valid fraction, class counts, E/eta/r resolutions, Gaussian and sumTS columns, loss)
python jobs/resolution/eval_table.py models/<run>/eval
# sigma(E)/E vs E with the sqrt(a^2/E + c^2) fit, CMS style (RES_FIT=peak|quant, RES_FIT_EMAX=<GeV>, RES_HIDE_ABOVE=1)
sbatch jobs/resolution/analysis_job.sh plot_resolution.py models/<run>/eval photons
# trackster-sum vs regression (sigma68 and median ratio vs E)
sbatch jobs/resolution/analysis_job.sh plot_sumts.py models/<run>/eval photons
```

Outputs: `models/<run>/eval/<set>/` (predictions, losses, plot chain, eval.log), `models/<run>/eval/gaussfit_summary.json`
and `models/<run>/eval/resolution_plots/`. Diagnostics used in issues #5-#7: `binning_scan.py` (fit stability vs
histogram binning), `epred_vs_sumts.py` (saturation check), `trackster_threshold_check.py` (events without a CLUE3D
trackster), `simtrackster_count.py` (SimTracksters per event). The resolution points are sigma(E_pred)/E_nominal from
a Gaussian fit to the E_pred histogram (peak-seeded core window, <= 10 % of events outside), the sigma68 columns are
quantile widths of E/E_true. The `evaluation-maskformer` Claude skill (`.claude/skills/`) walks through the whole flow
on both sites.

## Other scripts

| script | purpose |
|---|---|
| `hgcal/scripts/dataset/read_HGCDataset.py` | loads a parquet file through `HGCDataset` with the data block of a config (`--config`, default `configs/test/layercluster_simcluster.yaml`) and optionally plots the incidence matrices |
| `hgcal/scripts/plot/check_energy_scaling.py` | one-off diagnosis of the energy scaling of layer clusters with the rechit dict |
| `hgcal/scripts/jets/` | fastjet clustering example (`jets/instructions.md`) |

## Developer notes

- Both packages are found by `pyproject.toml` (`packages.find` under `src/`, `maskformer*` and `hgcal*`); the console
  script is `maskformer = hgcal.scripts.cli:cli`. The old top-level packages (`models`, `dataset`, `callbacks`,
  `utils`, `scripts`, `maskformers`) no longer exist: an image built before the split still ships them, which is why
  `train`/`evaluate` refuse legacy class paths instead of letting Python find the stale copies.
- `maskformer/scripts/lightning_cli.py` holds `MaskformerCLI`, hepattn's CLI plus the scale-dict path resolution and
  the legacy check; `maskformer/scripts/train.py` and `evaluate.py` take the experiment's `model_class` /
  `datamodule_class` (and a `cli_class`), `hgcal/scripts/train.py` and `evaluate.py` bind `MaskformerWrapper`,
  `HGCDataModule` and `HGCEvalCLI` (the plot chain). The HGCAL click group adds its own commands to the generic
  `train`/`evaluate` pieces of `maskformer/scripts/cli.py`.
- The prediction writers derive from `maskformer.callbacks.PredictionWriterBase` (output paths, un-scaling, loss
  summary, the parquet write) and implement `regression_transforms`, `reset_buffers`, `collect_batch` and
  `build_output`; `plot_chain_readable = False` tells the evaluation not to plot (the sectored writer).
- The plot modules insert their own folder into `sys.path` and import their siblings as top-level modules
  (`from hgcal_eval_common import ...`), so that `run_plot_chain.py` also runs by file path under another interpreter.
  Never import them package-qualified (`hgcal.scripts.plot.hgcal_eval_common`): there would be two module objects and
  `isinstance` checks between them would fail. `run_plot_chain` (imported by the CLI) is the one exception by design.
- Tests: `tests/test_cli.py`, `test_paths.py`, `test_configs.py`, `test_import_rule.py`, `test_migrate_config.py`,
  `test_scale_transform.py`, `test_plot_chain.py` and the SV ones `test_sv_features.py`, `test_sv_cli.py`,
  `test_sv_convert.py` (a synthetic ntuple written with uproot), `test_sv_plot_chain.py` run without torch
  (`PYTHONPATH=src:$PYTHONPATH python -m pytest tests/test_cli.py ...` in an LCG view); the rest, including
  `test_sv_dataset.py`, needs the container (`PYTHONPATH=src python -m pytest tests`).
