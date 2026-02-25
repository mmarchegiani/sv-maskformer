INITIAL_DIR=$(pwd)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEPATTN_DIR=~/ml4reco/hepattn
DATA_DIR=/home/export/mmarcheg/scratch/ml4reco/data_ticl/

export APPTAINERENV_PYTHONPATH=$BASE_DIR/src:$PYTHONPATH
cd $HEPATTN_DIR
apptainer exec --bind $DATA_DIR:/data --nv pixi.sif bash -c "pixi shell -e clic"
cd $INITIAL_DIR
