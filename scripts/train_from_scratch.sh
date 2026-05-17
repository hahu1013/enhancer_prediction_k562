#!/bin/bash
#SBATCH --job-name=enh_train_from_scratch
#SBATCH --partition=zhoulabq
#SBATCH --reservation=zhoulab
#SBATCH --nodelist=cri22cn416
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=18:00:00
#SBATCH --output=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.log
#SBATCH --error=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.err

# Phase 3 — from-scratch training condition (Briefing § 2.2 / § 4 Phase 3).
# Random PyTorch defaults; no Puffin-D warm-start. Baseline for the transfer
# comparison against the pretrained-init run.
#
# GPUs on cri22cn416 are NOT SLURM-managed (Gres=null). Do not use --gres;
# hardcode CUDA_VISIBLE_DEVICES instead. Check `nvidia-smi` on the node before
# launching to pick a free GPU.

set -eo pipefail

module load gcc/12.1.0
module load miniconda3/24.9.2

source activate /gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint

PYTHON=/gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint/bin/python
PROJECT=/gpfs/data/zhou-lab/haoyanghu/projects/enhancer_prediction

RUN_NAME=${RUN_NAME:-from_scratch}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/runs/${RUN_NAME}}
DATA_DIR=${DATA_DIR:-${PROJECT}/data/labels}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-5e-4}
POS_WEIGHT=${POS_WEIGHT:-25.0}
MIN_CALLABLE_FRACTION=${MIN_CALLABLE_FRACTION:-0.5}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-2500}
VAL_N_WINDOWS=${VAL_N_WINDOWS:-50}
LR_SCHEDULE=${LR_SCHEDULE:-plateau}
SEED=${SEED:-42}

# GPUs on cri22cn416 are NOT SLURM-managed. Hardcode the device.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# selene_mini available via the lab's conda env; no PYTHONPATH hack needed.

mkdir -p /gpfs/data/zhou-lab/haoyanghu/logs
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT}"

echo "[$(date -Iseconds)] Starting ${SLURM_JOB_NAME:-enh_train_from_scratch} on $(hostname)"
echo "  OUTPUT_DIR            = ${OUTPUT_DIR}"
echo "  DATA_DIR              = ${DATA_DIR}"
echo "  EPOCHS                = ${EPOCHS}"
echo "  STEPS_PER_EPOCH       = ${STEPS_PER_EPOCH}"
echo "  BATCH_SIZE            = ${BATCH_SIZE}"
echo "  LR                    = ${LR}"
echo "  POS_WEIGHT            = ${POS_WEIGHT}"
echo "  MIN_CALLABLE_FRACTION = ${MIN_CALLABLE_FRACTION}"
echo "  LR_SCHEDULE           = ${LR_SCHEDULE}"
echo "  SEED                  = ${SEED}"
echo "  CUDA_VISIBLE_DEVICES  = ${CUDA_VISIBLE_DEVICES}"

"${PYTHON}" scripts/train.py \
    --init from_scratch \
    --output-dir "${OUTPUT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --steps-per-epoch "${STEPS_PER_EPOCH}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --pos-weight "${POS_WEIGHT}" \
    --min-callable-fraction "${MIN_CALLABLE_FRACTION}" \
    --val-n-windows "${VAL_N_WINDOWS}" \
    --lr-schedule "${LR_SCHEDULE}" \
    --seed "${SEED}"

echo "[$(date -Iseconds)] Done with exit $?"
