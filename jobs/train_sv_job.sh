#!/bin/bash
#SBATCH --job-name=sv_maskformer_gv_tracks
#SBATCH --output=logs/slurm/sv_maskformer_gv_tracks_%j.out
#SBATCH --partition=work
#SBATCH --qos=light                # 7 days, max mps:25 per job
#SBATCH --time=7-00:00:00
#SBATCH --gres=mps:25
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Trains the SV MaskFormer (configs/sv/gv_tracks.yaml) on falcon, inside the CVMFS image, on the parquet files of
# `maskformer-sv convert --split` (train/ val/ test/ sub-folders of DATA_DIR).
# Usage: sbatch jobs/train_sv_job.sh [data_dir] [extra Lightning arguments...]
# Comet: set COMET_API_KEY in the environment, or add --trainer.logger.init_args.online false for an offline run.

BASE_PATH=/home/export/$USER/ml4reco/hgcal-offline-reco
DATA_DIR=${1:-/home/export/$USER/store/ml4reco/sv_reconstruction/parquet}
shift $(( $# < 1 ? $# : 1 ))

nvidia-smi
cd $BASE_PATH || exit 1
mkdir -p logs/slurm

APPTAINER_IMAGE=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/hgcal-ml4reco/hgcal-offline-reco:3.12-latest
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

apptainer exec \
  --nv \
  --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  --env COMET_API_KEY=${COMET_API_KEY:-} \
  --env LD_LIBRARY_PATH=/host-nvidia-libs:$LD_LIBRARY_PATH \
  -B /home/export/$USER \
  -B /cvmfs \
  -B /tmp \
  -B /usr/lib/x86_64-linux-gnu:/host-nvidia-libs \
  $APPTAINER_IMAGE \
  bash -c "
    export PYTHONPATH=$BASE_PATH/src:\$PYTHONPATH
    python -m sv train \
      --config configs/sv/gv_tracks.yaml \
      --data.train_path $DATA_DIR/train \
      --data.valid_path $DATA_DIR/val \
      --data.test_path $DATA_DIR/test \
      $*
  "
