#! /bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
source $DIR/load_env.sh
echo "Running from $DIR"

# SETUP PROJECT
PROJECT_NAME=thinkingearth
PROJECT_NAME_SHORT=ThinkEarth

# USER DIRS
USER_DIR=/gpfs/fs1/projects/nvr_dvl_research/users/$USER
PROJECT_USER_DIR=$USER_DIR/$PROJECT_NAME
RESULTS_DIR=$PROJECT_USER_DIR/train

# PROJECT DIRS
PROJECT_DIR=/gpfs/fs1/projects/nvr_dvl_research/users/$USER/thinkingearth
DATA_DIR=$PROJECT_DIR/data
PANGU_DIR=/home/dcg-adlr-fgerken-data/pangu

# DOCKER IMG
source $DIR/../ngc/submit.sh --push-only

CONTAINER_MOUNTS="$RESULTS_DIR:/workspace,"
CONTAINER_MOUNTS+="$DATA_DIR:/data,"
CONTAINER_MOUNTS+="$PANGU_DIR:/data/pangu"

# SLURM SETUP
PARTITIONS=backfill_dgx2h_m2,batch_dgx2h_m2,batch_dgx1_m2,backfill_dgx1_m2
ACCOUNT=nvr_dvl_research

# TRAIN RUN COMMMAND
NUM_WORKERS=4
NODES=1
GPUS=1

# NAME OF EXPERIMENT
EXP_NAME="$PROJECT_NAME_SHORT-Evaluation_2018_pangu_SURFACE_TEST5"

# RUN COMMANDconfigs
RUN_CMD="python pangu/evaluation/evaluate_2018.py "
RUN_CMD+="EXPERIMENT.SET_ID_FROM_SLURM_JOB_NAME True "
RUN_CMD+="SETUP.NUM_WORKERS ${NUM_WORKERS} "
RUN_CMD+="SETUP.N_NODES ${NODES} "
RUN_CMD+="SETUP.N_GPUS ${GPUS} "

# SLURM JOB SUBMIT
SLURM_ENV=OMP_NUM_THREADS=$OMP_NUM_THREADS

CMD="cd /pangu-weather;  $RUN_CMD"

ssh -t $USER@draco-rno-login.nvidia.com "submit_job -n $EXP_NAME --image $OCI_IMAGE --account $ACCOUNT --partition $PARTITIONS --nodes=$NODES --mounts $CONTAINER_MOUNTS --email_mode never --notify_on_start --notification_method slack --autoresume_timer 180 --duration 4.0 --tasks_per_node $GPUS --gpu $GPUS --setenv $SLURM_ENV --command ${CMD@Q}"
