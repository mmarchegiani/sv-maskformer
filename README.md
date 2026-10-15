# HGCAL Offline Reconstruction

Repository for HGCAL offline reconstruction using machine learning techniques. This repository contains code and tools for training and evaluating machine learning models on HGCAL simulated datasets.
A set of scripts to analyze the output and plot the clustering space is also included.

## Setup environment
The maskformer training code relies on a dedicated singularity container image with all the required python packages.
To open a shell within the singularity image, run the following command:
```bash
source setup_env.sh
```
In this environment is installed all the software needed for the training and evaluation of the models.

> [!WARNING]
> The singularity image in `/cvmfs` is systematically updated at every commit on the `master` branch of this repo.
> However, latencies might occur where the changes to this repo are not propagated yet to the singularity image in `/cvmfs`.
> If the singularity is outdated, a warning message will be displayed before entering in the singularity image.
> You can also display the dates of last image update and last commit with the `-v` argument: `source setup_env.sh -v`

## Download training and evaluation datasets
The training and evaluation datasets are stored on the CMU XRootD server (`root://cmsdata.phys.cmu.edu/`) under
`/store/group/hgcal-ml-reconstruction/training_datasets/`. To copy them locally, run:

```bash
bash fetch_datasets.sh [BASE_DIR]
```

`BASE_DIR` is the destination directory and defaults to `./data`. The script creates two subdirectories:

| directory | dataset | content |
|---|---|---|
| `$BASE_DIR/train` | `taus_noFineCalo_with_tracksters_50k` | 50k tau events, used for training and validation |
| `$BASE_DIR/eval` | `photons` | 1k single-photon events for each energy point (2, 3, 5, 10, 25, 50, 75, 100, 200, 300 GeV), used for evaluation |
| `$BASE_DIR/eval` | `pions` | 1k single-pion events for each energy point (2, 3, 5, 10, 25, 50, 75, 100, 200, 300 GeV), used for evaluation |
| `$BASE_DIR/eval` | `K0L` | 1k single-kaon events for each energy point (2, 3, 5, 10, 25, 50, 75, 100, 200, 300 GeV), used for evaluation |

All datasets are produced without fine calorimeter granularity (`noFineCalo`) and include the TICL trackster
collections. The files are copied with `xrdcp -f`, so files already present in `BASE_DIR` are overwritten. Since
the parquet files are large, point `BASE_DIR` to a location with enough space (e.g. your EOS or `/afs/.../work`
area rather than your AFS home), for example:

```bash
bash fetch_datasets.sh /eos/user/<u>/<username>/hgcal-datasets
```

Then set `data.train_path`/`data.valid_path`/`data.test_path` in your config to the downloaded files.
If the folder contains multiple files, it's possible to set `data.data_dir` in your config to pick all the files in the folder.
In addition, it's possible to specify the train, valid, test fractions, for example:
```yaml
  train_path: null
  valid_path: null
  test_path: null
  data_dir: /home/export/mmarcheg/store/ml4reco/data_maskformers/taus_noFineCalo_with_tracksters_50k
  split: [0.7, 0.15, 0.15]   # fractions, or file counts like [80, 10, 10]
  split_seed: 42           # shuffles the file list before cutting; null keeps rep0, rep1, ... order
```

## Input and target normalization

Every training config names **two** scale dicts (see `docs/NORMALIZATION_FIX.md` for the full status and rationale;
other bugs found while reviewing the repo are tracked in `docs/REVIEW_FINDINGS.md`):

| key in `data:` | fitted on | used by |
|---|---|---|
| `scale_dict_path` | the topocluster collection (`configs/hgcal_rechit_var_transform.yaml`, `configs/hgcal_layercluster_var_transform.yaml` or `configs/hgcal_trackster_clue3dhigh_var_transform.yaml`) | node input features in `HGCDataset` |
| `particle_scale_dict_path` | the particle collection (`configs/hgcal_mergedsimcluster_var_transform.yaml`, `configs/hgcal_simcluster_var_transform.yaml` or `configs/hgcal_simtrackster_var_transform.yaml`) | regression targets in `HGCDataset`, the regression task (`scale_dict_path: *particle_scale_dict_path`) and the prediction writer |

Both dicts use the keys `e`/`eta`/`r` but for populations whose spreads differ by orders of magnitude, so they must
stay separate files. To regenerate them for a (topocluster, particle) pair:

```bash
sbatch jobs/norm_par_job.sh                                          # RecHitHGC -> MergedSimCluster (configs/base.yaml)
TOPO=RecHitHGC   PART=SimCluster sbatch jobs/norm_par_job.sh         # configs/base_rechit_simcluster.yaml
TOPO=LayerCluster PART=SimCluster sbatch jobs/norm_par_job.sh        # configs/base_layercluster_simcluster.yaml
TOPO=ticlTrackstersCLUE3DHigh PART=ticlSimTracksters DATA_DIR=<dir> sbatch jobs/norm_par_job.sh   # configs/base_trackster_simtrackster.yaml
```

Each variable is first transformed (`sqrt`/`log`/`log1p`, set per variable in `TOPO_VAR_SPECS` / `PARTICLE_VAR_SPECS`
inside `src/scripts/dataset/compute_norm_par.py`), and only then are its `mean`/`std`/`min`/`max` measured and written
to the YAML — that order matters, because `hepattn`'s `VarTransform` applies the transform *before* shifting and
scaling. The job also writes before/after histograms to `plots/norm_par/` (`<Collection>_<var>_{before,after}.png`, e.g. `RecHitHGC_e_after.png`, `SimCluster_e_after.png`), and
fails if a variable's extreme lies further than its allowed number of standard deviations from the mean
(`NSIGMA_FAIL` in the script), so a broken variable is caught without tripping on physical tails.

For a quick check on a small subset (full runs must go through Slurm — the gateway node kills processes above 60% CPU):

```bash
python src/scripts/dataset/compute_norm_par.py --max-files 1 --max-events 20 --dry-run
python src/scripts/dataset/compute_norm_par.py --topocluster-collection LayerCluster --particle-collection SimCluster --max-files 1 --max-events 20 --dry-run
```

To check a fitted pair end to end through `HGCDataset` (scaled ranges of every input feature and target, finiteness,
and the specific failure modes the fix addressed):

```bash
python src/scripts/dataset/verify_norm.py --config configs/base_layercluster_simcluster.yaml --parquet <file.parquet> --num-events 200
```

## Training
To run the model training:

```bash
python src/scripts/train/train_maskformer.py fit --config configs/base.yaml
```

To skip sanity checking of the validation dataloader and test with a single worker, run the training with these options:
```bash
python src/scripts/train/train_maskformer.py fit --config configs/base.yaml --data.num_workers 0 --trainer.num_sanity_val_steps 0
```

Note that the parameters in `configs/base.yaml` are a template and need to be tuned for the HGCAL reconstruction case.

### Checkpoints and early stopping

Checkpoints go to `logs/<run>/ckpts/`. hepattn's `Checkpoint` callback writes every epoch as
`epoch=NNN-val_loss=X.XXXXX.ckpt`; the `BestCheckpoint` callback (`src/callbacks/best_checkpoint.py`) keeps, next to
them, a single copy of the best epoch so far as `best-epoch=NNN-val_loss=X.XXXXX.ckpt`, replacing the previous
`best-*` file whenever a later epoch lowers `val/loss`.

The `EarlyStopping` callback (`src/callbacks/early_stopping.py`, a thin layer over Lightning's) ends the training
before `trainer.max_epochs` when the total validation loss `val/loss` has not decreased for `patience` consecutive
epochs (a NaN/inf `val/loss` also stops it). Both are enabled in every training config in `configs/`:

```yaml
trainer:
  callbacks:
    - class_path: callbacks.best_checkpoint.BestCheckpoint
      init_args:
        monitor: val/loss
    - class_path: callbacks.early_stopping.EarlyStopping
      init_args:
        patience: 25   # epochs without improvement before stopping
```

Remove the `EarlyStopping` entry to always train for `max_epochs`.


## Customize topocluster and particle collections
Multiple (topocluster, particle) combinations are supported:
- (`RecHitHGC`, `SimCluster`) 
- (`RecHitHGC`, `MergedSimCluster`) 
- (`LayerCluster`, `SimCluster`)
- (`ticlTrackstersCLUE3DHigh`, `ticlSimTracksters`) — TICLv5 tracksters and SimTracksters, see below

To customize the collections, specify the collections in the YAML config file in the `data` section.
For example, to train a reconstruction model using `LayerClusters` as topoclusters and `SimClusters` as true particles, one should specify:

```yaml
data:
  topocluster_collection: LayerCluster
  particle_collection: SimCluster

  inputs:
    topocluster:
      - "e"
      - "eta"
      - "cosphi"
      - "sinphi"
      - "r"

  targets:
    particle:
      - e
      - eta
      - cosphi
      - sinphi
      - r

  # scale dicts fitted on THESE collections (see "Input and target normalization")
  scale_dict_path: &scale_dict_path configs/hgcal_layercluster_var_transform.yaml
  particle_scale_dict_path: &particle_scale_dict_path configs/hgcal_simcluster_var_transform.yaml
```

The two scale dicts must match the chosen collections: the node dict is fitted on the topocluster collection
and the particle dict on the particle collection, and the regression task must reference
`*particle_scale_dict_path`. `configs/base_layercluster_simcluster.yaml` is a complete example.

### Tracksters as nodes, SimTracksters as particles

`configs/base_trackster_simtrackster.yaml` trains on TICLv5 CLUE3D tracksters (`ticlTrackstersCLUE3DHigh`) to
predict SimTracksters seeded by SimClusters (`ticlSimTracksters`). It needs a parquet file produced from the TICL
NanoAOD tables (CMSSW_20_0_0_pre1 or later, e.g. `taus_fineCalo_split.parquet`) that contains, besides the two
collections:

- the reco → sim **association table** (`CLUE3DHighToSimTSByHits`, or `CLUE3DHighToSimTSByLCs`), written in
  CMSSW by `AllTracksterToSimTracksterAssociatorsByHitsProducer` and flattened by the NanoAOD
  `AssociationOneToManyFlatTableProducer`. Per trackster it lists the SimTracksters it shares energy with
  (`...Links_index`), the shared energy in GeV (`...Links_sharedEnergy`) and the association score
  (`...Links_score`, lower is better). Select the table with `data.association_collection`.
- the `SimCluster` collection: SimTracksters carry no `pdgId`, so it (and `eventId`/`bunchCrossing`) is read off
  the seed SimCluster through `ticlSimTracksters_seedIndex`.

Unlike the hit and layer-cluster pairs, where every node is assigned to a single best-match particle, the
incidence matrix here is **soft**: entry (particle, node) is the fraction of the trackster's energy shared with
that SimTrackster (the shares of one trackster sum to at most 1, the rest being energy without simulated
contribution), so a trackster that merges two showers is split between them. `data.min_link_fraction` drops links below a given share, `data.incidence_cutval` decides which
shares count as a node of the particle for the mask target, and `data.normalize_incidence: false` keeps the
energy-share normalisation (the per-particle row normalisation of the other configs would destroy it). The
particle position target is the SimTrackster barycenter and the energy target its `regressed_energy`, the seed's
true energy at the calorimeter boundary (the best achievable energy reconstruction; the layer-cluster sum
`raw_energy` is not used). The trackster energy input is `raw_energy`, on purpose: the model must not depend on the
TICL energy-regression DNN. The `boundary*` branches are available in the file if another position target is
preferred (`PARTICLE_FIELD_MAP` in `src/dataset/collections.py`, the torch-free module holding the per-collection
branch names and truth-association tables shared by the dataset and the evaluation plots).

### Non-finite (NaN/inf) guards

A NaN in a single event turns the epoch-mean `val/loss` into NaN while the on-step `train/loss` (logged every
`log_every_n_steps` batches) can still look fine, and an L1 regression term with a NaN target even keeps finite
gradients, so such a run trains on silently. The pipeline therefore asserts finiteness where a bad value can
enter (`src/dataset/finite_checks.py`):

- `HGCDataset`: the scale dicts (finite shift, non-zero scale), every raw branch of the pair once per file, the
  domain of the `log`/`log1p` energy transforms, the energy shares of the TICL association table (finite, in
  [0, 1], summing to at most 1 per trackster), and per event the scaled node inputs, the scaled particle targets
  and the incidence matrix (finite, in [0, 1], node columns or particle rows summing to at most 1).
- `HGCMaskformer`: every batch on arrival (valid nodes, valid particles, the dense incidence target) and every
  loss term after `model.loss`. The first non-finite loss raises an `AssertionError` naming the terms and the
  event numbers, after saving the batch (`inputs`, `targets`, `losses`) to
  `<log_dir>/nonfinite_<stage>_step<N>_batch<M>.pt` (or `--model.nonfinite_dump_dir`) so it can be replayed on a CPU.

`tests/test_finite_checks.py` exercises the checks on synthetic tensors.

## Evaluation
To evaluate a trained MaskFormer model, run inference on a test file and write out predictions:

```bash
python src/scripts/evaluate/evaluate_maskformer.py test \
    --config configs/test.yaml \
    --data.test_path /path/to/test.parquet \
    --trainer.precision 32-true \
    --matmul_precision highest
```

- Use `configs/test.yaml` for the (`RecHitHGC`, `MergedSimCluster`) collections, or `configs/test_ticl_collections.yaml` for (`LayerCluster`, `SimCluster`) — pick whichever matches the collections the model was trained on.
- If `--ckpt_path` is omitted, the best checkpoint (lowest `val/loss`) is picked up automatically from a `ckpts/` directory next to the `--config` file. Otherwise pass it explicitly, e.g. `--ckpt_path logs/<run>/ckpts/epoch=160-val_loss=6.15737.ckpt` (or the `best-epoch=...` copy written by `BestCheckpoint`, see [Checkpoints and early stopping](#checkpoints-and-early-stopping)).
- Output: a parquet file written next to the checkpoint, `<ckpt_name>__test.parquet`, with the same schema as the input test file plus prediction branches (see the `HGCPredictionWriter` docstring in `src/callbacks/prediction_writer.py` for the full list of added fields), and a `<ckpt_name>__test_losses.json` with the mean test loss per task.
- Plots: after the test loop the script runs the standard plotting chain (`src/scripts/plot/run_plot_chain.py`) on the
  prediction file and writes the figures next to it, in `<ckpt_name>__test_plots/`:
  - `resolution/`, `resolution_by_class/`, `efficiency/`, `diagnostics/`: per-cluster resolutions and clustering
    efficiency, plus `metrics.json`/`summary.txt`;
  - `cluster_properties/`: true vs predicted distribution of every regressed feature (e, eta, cosphi, sinphi, r),
    `response/` the 1D response `pred/true` per feature and `response/by_<feat>/` the response of every feature in
    bins of every (true) feature;
  - `topocluster_features/raw/` and `topocluster_features/transform/`: the node input features (`data.inputs.topocluster`)
    as read from the file and after the scale-dict transform the dataset applies (`data.scale_dict_path`);
  - `event_reconstruction/`: the event as one object, i.e. the sum of all predicted clusters (energy sum,
    energy-weighted eta/cosphi/sinphi/r) against the true particles - what matters for a photon reconstructed as
    e+e-. Only matched queries enter (the kinematics of fake queries are not stored; their count is plotted);
  - `event_display/`: 3D views and 2D projections (y-x, x-z, y-z, rho-z) of the first events, true and predicted
    clusters as rings sized by their energy, nodes as dots;
  - `energy/`: the energy deep-dive with `energy_metrics.json`/`energy_summary.txt`.

  `--plot_dir <dir>` writes them elsewhere, `--skip_plots true` skips them, `--plot_max_events <N>` plots only the
  first N events, `--plot_n_display <N>` sets the number of event displays (default 5). The chain runs as a
  subprocess of the same interpreter (set `HGCAL_PLOT_PYTHON` to use another one, e.g. one with `mplhep`); if it
  fails, the evaluation still succeeds and the command to rerun it by hand is printed.

### Plotting a prediction file by hand
The chain needs only awkward, numpy, pyarrow and matplotlib (`mplhep` is optional; on lxplus the LCG view
`source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh` has all of them):

```bash
python src/scripts/plot/run_plot_chain.py --input <ckpt_dir>/<ckpt_name>__test.parquet --outdir plots/<run> \
    --scale-dict configs/hgcal_trackster_clue3dhigh_var_transform.yaml --node-features e eta cosphi sinphi r
```

The (topocluster, particle) collections are auto-detected from the file (the ones carrying `*_pred_match_idx` and
`*_pred_valid`); every pair the dataset supports is handled, including the trackster/SimTrackster pair whose truth
comes from the TICL association table. Options: `--topocluster-collection`/`--particle-collection` to force the
collections, `--association-collection` and `--min-link-fraction` for the trackster pair (the evaluation script
passes the model's own settings), `--max-events N` for a subset, `--skip-energy` to skip the energy deep-dive,
`--scale-dict` (the node scale dict of the training config, `data.scale_dict_path`) to get the transformed node
features (without it only `topocluster_features/raw/` is written), `--node-features` (the model's
`data.inputs.topocluster`, default `e eta cosphi sinphi r`), `--n-display N` / `--skip-display` for the event
displays. `src/scripts/plot/run_all.sh <parquet> [outdir]` wraps the same command; the modules it runs
(`plot_maskformer_eval.py`, `energy_investigation.py`, `cluster_properties.py`, `topocluster_features.py`,
`event_reconstruction.py`, `event_display.py`) can also be run on their own with `--input`/`--outdir`.
