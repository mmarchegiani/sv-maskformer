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

## Input and target normalization

Every training config names **two** scale dicts (see `docs/NORMALIZATION_FIX.md` for the full status and rationale;
other bugs found while reviewing the repo are tracked in `docs/REVIEW_FINDINGS.md`):

| key in `data:` | fitted on | used by |
|---|---|---|
| `scale_dict_path` | the topocluster collection (`configs/hgcal_rechit_var_transform.yaml` or `configs/hgcal_layercluster_var_transform.yaml`) | node input features in `HGCDataset` |
| `particle_scale_dict_path` | the particle collection (`configs/hgcal_mergedsimcluster_var_transform.yaml` or `configs/hgcal_simcluster_var_transform.yaml`) | regression targets in `HGCDataset`, the regression task (`scale_dict_path: *particle_scale_dict_path`) and the prediction writer |

Both dicts use the keys `e`/`eta`/`r` but for populations whose spreads differ by orders of magnitude, so they must
stay separate files. To regenerate them for a (topocluster, particle) pair:

```bash
sbatch jobs/norm_par_job.sh                                          # RecHitHGC -> MergedSimCluster (configs/base.yaml)
TOPO=RecHitHGC   PART=SimCluster sbatch jobs/norm_par_job.sh         # configs/base_rechit_simcluster.yaml
TOPO=LayerCluster PART=SimCluster sbatch jobs/norm_par_job.sh        # configs/base_layercluster_simcluster.yaml
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


## Customize topocluster and particle collections
Multiple (topocluster, particle) combinations are supported:
- (`RecHitHGC`, `SimCluster`) 
- (`RecHitHGC`, `MergedSimCluster`) 
- (`LayerCluster`, `SimCluster`)

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
- If `--ckpt_path` is omitted, the best checkpoint (lowest `val/loss`) is picked up automatically from a `ckpts/` directory next to the `--config` file. Otherwise pass it explicitly, e.g. `--ckpt_path logs/<run>/ckpts/epoch=160-val_loss=6.15737.ckpt`.
- Output: a parquet file written next to the checkpoint, `<ckpt_name>__test.parquet`, with the same schema as the input test file plus prediction branches (see the `HGCPredictionWriter` docstring in `src/callbacks/prediction_writer.py` for the full list of added fields), and a `<ckpt_name>__test_losses.json` with the mean test loss per task.
