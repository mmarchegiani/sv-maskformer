#!/bin/bash
# Trains the SV MaskFormer on an lxplus-gpu machine (no Slurm) inside the CVMFS image, with EOS reachable through the
# caller's Kerberos ticket (kinit first). The committed config reads the parquet files on EOS
# (/eos/cms/store/group/phys_btag/mmarcheg/sv_reconstruction/training_datasets/TTto4Q_tracks_GenVertex_50k/parquet/{train,val,test}); override with --data.train_path etc.
#
# Usage (in tmux/screen): bash jobs/train_sv_lxplus.sh [config] [extra Lightning arguments...]
#   bash jobs/train_sv_lxplus.sh configs/sv/gv_tracks.yaml --trainer.logger.init_args.online false --attn-type torch
# `--attn-type torch` is needed on the T4s of lxplus-gpu (no flash attention); the A100/V100 nodes can keep flash-varlen.

BASE_PATH=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
CONFIG=${1:-configs/sv/gv_tracks.yaml}
shift $(( $# < 1 ? $# : 1 ))
APPTAINER_IMAGE=${APPTAINER_IMAGE:-/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/hgcal-ml4reco/hgcal-offline-reco:3.12-latest}
cd "$BASE_PATH" || exit 1

nvidia-smi || { echo "ERROR: no GPU visible; run this on lxplus-gpu.cern.ch" >&2; exit 1; }
KRB5CC=${KRB5CCNAME#FILE:}
if [ -z "$KRB5CCNAME" ] || [ "$KRB5CC" = "$KRB5CCNAME" ]; then
  KRB5CC=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/krb5cc
fi
[ -f "$KRB5CC" ] || { echo "ERROR: Kerberos cache $KRB5CC not found; run kinit" >&2; exit 1; }

export APPTAINERENV_PYTHONPATH="$BASE_PATH/src"
export APPTAINERENV_HGCAL_REPO_ROOT="$BASE_PATH"
export APPTAINERENV_KRB5CCNAME="FILE:$KRB5CC"
export APPTAINERENV_COMET_API_KEY="${COMET_API_KEY:-}"
apptainer exec --nv \
  -B /cvmfs -B /tmp -B "/eos/user/${USER:0:1}/$USER" -B /eos/cms/store/group/phys_btag -B "$BASE_PATH" -B /etc/sysconfig/ngbauth-submit -B "$(dirname "$KRB5CC")" \
  --pwd "$BASE_PATH" "$APPTAINER_IMAGE" \
  python -m sv train --config "$CONFIG" "$@"
