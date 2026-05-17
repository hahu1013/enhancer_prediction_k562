#!/bin/bash
#SBATCH --job-name=enh_run_ism
#SBATCH --partition=zhoulabq
#SBATCH --reservation=zhoulab
#SBATCH --nodelist=cri22cn416
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.log
#SBATCH --error=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.err

# Phase 6.1 — ISM on top-K chr10 positions. GPU job (the bulk of the time
# is alt-sequence forward passes). No --gres; hardcode CUDA_VISIBLE_DEVICES.
# Module loads: gcc + miniconda3 only (durable rule).

set -eo pipefail

module load gcc/12.1.0
module load miniconda3/24.9.2
source activate /gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint

PYTHON=/gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint/bin/python
PROJECT=/gpfs/data/zhou-lab/haoyanghu/projects/enhancer_prediction

CHECKPOINT=${CHECKPOINT:-${PROJECT}/runs/pretrained/best.pt}
PREDICTIONS=${PREDICTIONS:-${PROJECT}/runs/pretrained/pred_chr10_best.npz}
OUTPUT=${OUTPUT:-${PROJECT}/interpretation/ism_saliency_chr10.npz}
CHROM=${CHROM:-chr10}
TOP_K=${TOP_K:-1000}
MIN_SEPARATION=${MIN_SEPARATION:-200}
WINDOW_HALF=${WINDOW_HALF:-25}
BATCH_SIZE=${BATCH_SIZE:-64}
SEED=${SEED:-42}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p /gpfs/data/zhou-lab/haoyanghu/logs
mkdir -p "$(dirname "${OUTPUT}")"
cd "${PROJECT}"

echo "[$(date -Iseconds)] Starting ${SLURM_JOB_NAME:-enh_run_ism} on $(hostname)"
echo "  CHECKPOINT           = ${CHECKPOINT}"
echo "  PREDICTIONS          = ${PREDICTIONS}"
echo "  OUTPUT               = ${OUTPUT}"
echo "  CHROM                = ${CHROM}"
echo "  TOP_K                = ${TOP_K}"
echo "  MIN_SEPARATION       = ${MIN_SEPARATION}"
echo "  WINDOW_HALF          = ${WINDOW_HALF}"
echo "  BATCH_SIZE           = ${BATCH_SIZE}"
echo "  SEED                 = ${SEED}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"

"${PYTHON}" scripts/run_ism.py \
    --checkpoint "${CHECKPOINT}" \
    --predictions "${PREDICTIONS}" \
    --output "${OUTPUT}" \
    --chrom "${CHROM}" \
    --top-k "${TOP_K}" \
    --min-separation "${MIN_SEPARATION}" \
    --window-half "${WINDOW_HALF}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}"

echo "[$(date -Iseconds)] Done with exit $?"
