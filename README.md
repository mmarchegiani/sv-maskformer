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

```

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
