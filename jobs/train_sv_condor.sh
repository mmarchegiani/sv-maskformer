#!/bin/bash
# HTCondor job (CERN GPU batch, submitted from lxplus with jobs/train_sv_condor.sub): trains the SV MaskFormer inside
# the CVMFS image, evaluates the best checkpoint on the test split and publishes the plots on the EOS web area, all
# unattended. EOS and AFS are reached through the Kerberos credential condor delegates to the job
# (MY.SendCredential = True in the submit file), so no session has to stay open.
#
#   condor_submit jobs/train_sv_condor.sub                        # full training, configs/sv/gv_tracks.yaml
#   condor_submit jobs/train_sv_condor.sub smoke=1 flavour=longlunch   # 3-batch validation of the whole chain
#   condor_submit jobs/train_sv_condor.sub name=sv_gv_tracks_v2 extra="--max-epochs 20"
#   condor_submit jobs/train_sv_condor.sub run_dir=<runs_root>/job_<cluster>/<run> flavour=longlunch   # evaluate only
#   condor_submit jobs/train_sv_condor.sub config=<config> name=<name> resume=<runs_root>/job_<cluster>/<run>   # resume
#
# Arguments (set by the submit file): <repo> <config> <name> <runs_root> <www_root> <smoke 0/1> <cluster id>
# <run_dir or -> <resume or -> [extra Lightning arguments...]; a run_dir skips the training and evaluates that run.
# resume continues a training that stopped (wall time, eviction, crash) from a checkpoint: a .ckpt file, or a run
# folder / its ckpts/ folder, of which the last epoch=NNN-*.ckpt is taken. Pass the config and name of the stopped
# run. Epoch, optimizer, scheduler and early-stopping state are restored; the run continues in a new folder
# job_<new cluster>/<name>_<new timestamp>/ (and a new Comet experiment), so its best-*.ckpt, which the evaluation
# uses, is the best of the resumed epochs only. Outputs:
#   <runs_root>/job_<cluster>/<name>_<timestamp>/        config.yaml, ckpts/ (last + best), dataset counts, the
#                                                        prediction file and losses next to the best checkpoint,
#                                                        STATUS (RUNNING / TRAIN_FAILED / EVAL_FAILED / DONE / KILLED)
#   <www_root>/<name>_<timestamp>/evaluation/test/      the plots, metrics.json, summary.txt (web index copied)
#   logs/condor/<name>_<cluster>.{out,err,log}           condor stdout / stderr / event log, in the repo
# When condor stops the job (wall time of the flavour exceeded, condor_rm, eviction) it sends SIGTERM: the script
# then writes KILLED, the stage it was in and the resume= value of the last checkpoint to STATUS, and stops the
# container.
# On an A100 / H100 / H200 (compute capability >= 8) the config's flash attention and bf16 are used; on a T4 the
# script falls back to the torch attention backend and fp32 (no bf16 there, fp16 can overflow). The image's PyTorch
# has no kernels for the V100s (capability 7.0), which the submit file excludes.
# Comet: the job reads COMET_API_KEY from the `export COMET_API_KEY=...` line of ~/.bashrc on AFS (COMET_KEY_FILE to
# use another file); without a key the logger runs offline.

set -u

BASE_PATH=${1:?repo}; CONFIG=${2:?config}; NAME=${3:?name}; RUNS_ROOT=${4:?runs root}; WWW_ROOT=${5:?www root}
SMOKE=${6:-0}; CLUSTER=${7:-manual}; RUN_DIR_GIVEN=${8:--}
# The 9th argument is the resume checkpoint; jobs queued by a submit file without it have the extra arguments there
RESUME=-
if [ $# -ge 9 ] && [[ "$9" != -?* ]]; then RESUME=$9; shift 9; else shift $(( $# < 8 ? $# : 8 )); fi
EXTRA_ARGS=("$@")
APPTAINER_IMAGE=${APPTAINER_IMAGE:-/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/hgcal-ml4reco/hgcal-offline-reco:3.12-latest}
PB_COPY_INDEX=${PB_COPY_INDEX:-/eos/user/${USER:0:1}/$USER/www/php-plots/bin/pb_copy_index.py}
ROOT=$RUNS_ROOT/job_$CLUSTER

echo "=== SV MaskFormer condor job: cluster $CLUSTER on $(hostname), $(date)"
echo "repo $BASE_PATH | config $CONFIG | name $NAME | smoke $SMOKE | runs $ROOT | plots $WWW_ROOT | run_dir $RUN_DIR_GIVEN | resume $RESUME | extra: ${EXTRA_ARGS[*]:-}"
cd "$BASE_PATH" || { echo "ERROR: cannot enter $BASE_PATH (AFS credential?)" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "ERROR: $CONFIG not found" >&2; exit 1; }

# ---- GPU: flash attention + bf16 on Ampere or newer, torch attention + fp32 otherwise ------------------------------
nvidia-smi || { echo "ERROR: no GPU visible" >&2; exit 1; }
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_ARGS=()
case "$GPU_NAME" in
  *A100*|*H100*|*H200*|*A30*|*A40*|*L40*|*L4*) echo "GPU: $GPU_NAME (flash-varlen, bf16-mixed)" ;;
  *) echo "GPU: $GPU_NAME: no flash attention / bf16, using the torch backend and 32-true"
     GPU_ARGS=(--model.model.init_args.encoder.init_args.attn_type torch --trainer.precision 32-true) ;;
esac

# ---- Kerberos cache for EOS inside the container (condor delegates a FILE: cache) ----------------------------------
KRB5CC=${KRB5CCNAME#FILE:}
if [ -z "${KRB5CCNAME:-}" ] || [ "$KRB5CC" = "$KRB5CCNAME" ]; then
  KRB5CC=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/krb5cc
fi
[ -f "$KRB5CC" ] || { echo "ERROR: Kerberos cache $KRB5CC not found (MY.SendCredential = True in the submit file?)" >&2; exit 1; }
klist -s -c "FILE:$KRB5CC" || { echo "ERROR: Kerberos ticket in $KRB5CC is not valid" >&2; exit 1; }
mkdir -p "$ROOT" "$WWW_ROOT" || { echo "ERROR: cannot create $ROOT / $WWW_ROOT on EOS" >&2; exit 1; }
echo "RUNNING $(date) on $(hostname), GPU $GPU_NAME" > "$ROOT/STATUS"

# ---- KILLED status when condor stops the job ------------------------------------------------------------------------
# condor sends SIGTERM (then SIGKILL after a grace period) when it removes or evicts the job. bash runs a trap only
# once the foreground command returns, so the container runs in the background and the script waits for it.
STAGE=setup; CHILD=
on_kill() {
  trap '' TERM INT
  echo "KILLED by SIG$1 during $STAGE $(date) on $(hostname) (condor stopped the job: wall time limit, condor_rm or eviction)" > "$ROOT/STATUS"
  echo "ERROR: SIG$1 received during $STAGE, $(date): STATUS set to KILLED" >&2
  [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null
  if [ "$STAGE" = training ]; then
    LAST=$(ls "$ROOT"/"$NAME"_*/ckpts/epoch=*.ckpt 2>/dev/null | sort | tail -1)
    [ -n "$LAST" ] && echo "last checkpoint: $LAST (continue with resume=$LAST)" >> "$ROOT/STATUS"
  fi
  [ -n "$CHILD" ] && wait "$CHILD" 2>/dev/null
  exit 143
}
trap 'on_kill TERM' TERM
trap 'on_kill INT' INT

BINDINGS=(-B /cvmfs -B /tmp -B "/eos/user/${USER:0:1}/$USER" -B /eos/cms/store/group/phys_btag -B "$BASE_PATH" -B "$(dirname "$KRB5CC")")
[ -n "${_CONDOR_SCRATCH_DIR:-}" ] && BINDINGS+=(-B "$_CONDOR_SCRATCH_DIR")
[ -e /etc/sysconfig/ngbauth-submit ] && BINDINGS+=(-B /etc/sysconfig/ngbauth-submit)
export APPTAINERENV_PYTHONPATH="$BASE_PATH/src"
export APPTAINERENV_HGCAL_REPO_ROOT="$BASE_PATH"
export APPTAINERENV_KRB5CCNAME="FILE:$KRB5CC"
# Comet key: condor does not carry the submit shell's environment, and `getenv` would expose the key in the job
# ClassAd (condor_q -l), so it is read from the `export COMET_API_KEY=...` line of ~/.bashrc on AFS instead.
BASHRC=${COMET_KEY_FILE:-/afs/cern.ch/user/${USER:0:1}/$USER/.bashrc}
if [ -z "${COMET_API_KEY:-}" ] && [ -r "$BASHRC" ]; then
  COMET_API_KEY=$(sed -nE "s/^[[:space:]]*export[[:space:]]+COMET_API_KEY=[\"']?([^\"'[:space:]#]+).*/\1/p" "$BASHRC" | tail -1)
fi
LOGGER_ARGS=()
if [ -n "${COMET_API_KEY:-}" ]; then
  export APPTAINERENV_COMET_API_KEY="$COMET_API_KEY"
  echo "Comet: API key found, logging online"
else
  echo "WARNING: no COMET_API_KEY (environment or $BASHRC): the Comet logger runs offline"
  LOGGER_ARGS=(--trainer.logger.init_args.online false)
fi
run_in_container() {
  apptainer exec --nv "${BINDINGS[@]}" --pwd "$BASE_PATH" "$APPTAINER_IMAGE" "$@" &
  CHILD=$!
  wait "$CHILD"; local rc=$?
  CHILD=
  return $rc
}

# ---- training (skipped with a run_dir) ----------------------------------------------------------------------------
SMOKE_TRAIN=(); SMOKE_EVAL=()
if [ "$SMOKE" != "0" ]; then
  SMOKE_TRAIN=(--max-epochs 1 --trainer.limit_train_batches 3 --trainer.limit_val_batches 2 --data.num_train 256 --data.num_val 64)
  SMOKE_EVAL=(--data.num_test 32 --plot-max-events 32)
fi
if [ "$RUN_DIR_GIVEN" = "-" ]; then
  RESUME_ARGS=()
  if [ "$RESUME" != "-" ]; then
    CKPT=${RESUME%/}
    if [ -d "$CKPT" ]; then  # a run folder or its ckpts/: the last epoch (zero-padded numbers, so the sort is by epoch)
      [ -d "$CKPT/ckpts" ] && CKPT=$CKPT/ckpts
      CKPT=$(ls "$CKPT"/epoch=*.ckpt 2>/dev/null | sort | tail -1)
    fi
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
      echo "TRAIN_FAILED resume checkpoint not found: $RESUME $(date)" > "$ROOT/STATUS"
      echo "ERROR: resume checkpoint not found: $RESUME (a .ckpt file, a run folder or its ckpts/ folder)" >&2
      exit 1
    fi
    RESUME_ARGS=(--ckpt-path "$CKPT")
    echo "RUNNING $(date) on $(hostname), GPU $GPU_NAME, resumed from $CKPT" > "$ROOT/STATUS"
    echo "=== resuming from $CKPT"
  fi
  STAGE=training
  echo "=== training: $(date)"
  run_in_container python -m sv train --config "$CONFIG" --name "$NAME" --trainer.default_root_dir "$ROOT" \
    --data.num_workers 3 "${RESUME_ARGS[@]}" "${GPU_ARGS[@]}" "${LOGGER_ARGS[@]}" "${SMOKE_TRAIN[@]}" "${EXTRA_ARGS[@]}"
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "TRAIN_FAILED rc=$RC $(date)" > "$ROOT/STATUS"
    echo "ERROR: training exited with $RC" >&2
    exit $RC
  fi
  RUN_DIR=$(ls -d "$ROOT"/"$NAME"_*/ 2>/dev/null | head -1)
  RUN_DIR=${RUN_DIR%/}
else
  echo "=== evaluation only of $RUN_DIR_GIVEN"
  RUN_DIR=${RUN_DIR_GIVEN%/}
fi
if [ -z "$RUN_DIR" ] || [ ! -f "$RUN_DIR/config.yaml" ] || ! ls "$RUN_DIR"/ckpts/*.ckpt >/dev/null 2>&1; then
  echo "TRAIN_FAILED no run folder with config.yaml and checkpoints under $ROOT $(date)" > "$ROOT/STATUS"
  echo "ERROR: no run folder with config.yaml and checkpoints under $ROOT" >&2
  exit 1
fi
echo "=== trained: $RUN_DIR ($(ls "$RUN_DIR"/ckpts/*.ckpt | wc -l) checkpoints), $(date)"

# ---- evaluation of the best checkpoint on the config's test split, plots on the EOS web area ----------------------
# The torch attention backend: the evaluation runs in fp32 (the CLI's default precision) and flash attention needs
# bf16 ("mat1 and mat2 must have the same dtype" inside the compiled model otherwise); same maths, as on the T4s.
PLOT_DIR=$WWW_ROOT/$(basename "$RUN_DIR")/evaluation/test
mkdir -p "$PLOT_DIR"
STAGE=evaluation
echo "=== evaluation -> $PLOT_DIR: $(date)"
run_in_container python -m sv evaluate -c "$RUN_DIR/config.yaml" --root-dir "$RUN_DIR" --plot-dir "$PLOT_DIR" \
  --num-workers 0 --attn-type torch "${SMOKE_EVAL[@]}"
RC=$?
if [ $RC -ne 0 ]; then
  echo "EVAL_FAILED rc=$RC run=$RUN_DIR $(date)" > "$ROOT/STATUS"
  echo "ERROR: evaluation exited with $RC" >&2
  exit $RC
fi
if [ -x "$PB_COPY_INDEX" ]; then
  (cd "$WWW_ROOT/$(basename "$RUN_DIR")" && python3 "$PB_COPY_INDEX" -r .) || echo "WARNING: $PB_COPY_INDEX failed; the plots are there without the web index"
  [ -f "$WWW_ROOT/index.php" ] || cp "$WWW_ROOT/../maskformers/index.php" "$WWW_ROOT/" 2>/dev/null || true
else
  echo "WARNING: $PB_COPY_INDEX not found; the plots are there without the web index"
fi
echo "DONE run=$RUN_DIR plots=$PLOT_DIR $(date)" > "$ROOT/STATUS"
echo "=== done: $(date)"
echo "run:   $RUN_DIR"
echo "plots: $PLOT_DIR"
[ -f "$PLOT_DIR/summary.txt" ] && { echo "--- summary.txt"; cat "$PLOT_DIR/summary.txt"; }
exit 0
