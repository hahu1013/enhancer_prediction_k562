#!/bin/bash
#SBATCH --job-name=enh_run_inference
#SBATCH --partition=zhoulabq
#SBATCH --reservation=zhoulab
#SBATCH --nodelist=cri22cn416
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.log
#SBATCH --error=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.err

# Phase 4 — tiled inference for the K562 enhancer model. GPU sbatch (no
# --gres because GPUs on cri22cn416 are not SLURM-managed; hardcode
# CUDA_VISIBLE_DEVICES instead). The same wrapper runs for all four
# checkpoints — set CHECKPOINT and OUTPUT env vars, e.g.:
#
#   CHECKPOINT=runs/from_scratch/best.pt \
#   OUTPUT=runs/from_scratch/pred_chr10_best.npz \
#       sbatch scripts/run_inference.sh
#
# Inference is fast — one chromosome sweep is a few minutes, not hours.

set -eo pipefail

module load gcc/12.1.0
module load miniconda3/24.9.2
source activate /gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint

PYTHON=/gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint/bin/python
PROJECT=/gpfs/data/zhou-lab/haoyanghu/projects/enhancer_prediction

# Required: which checkpoint + where to write the .npz.
CHECKPOINT=${CHECKPOINT:?CHECKPOINT env var is required (path to a .pt file)}
OUTPUT=${OUTPUT:?OUTPUT env var is required (path to write the .npz)}

# Optional overrides — defaults match the Phase 4+5 briefing.
CHROM=${CHROM:-chr10}
DATA_DIR=${DATA_DIR:-${PROJECT}/data/labels}
GENOME_PATH=${GENOME_PATH:-/gpfs/data/zhou-lab/haoyanghu/data/reference/hg38.fa.gz}
STRIDE=${STRIDE:-50000}
WINDOW_SIZE=${WINDOW_SIZE:-100000}
BATCH_SIZE=${BATCH_SIZE:-8}

# GPUs on cri22cn416 are NOT SLURM-managed. Hardcode the device.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p /gpfs/data/zhou-lab/haoyanghu/logs
mkdir -p "$(dirname "${OUTPUT}")"
cd "${PROJECT}"

echo "[$(date -Iseconds)] Starting ${SLURM_JOB_NAME:-enh_run_inference} on $(hostname)"
echo "  CHECKPOINT           = ${CHECKPOINT}"
echo "  OUTPUT               = ${OUTPUT}"
echo "  CHROM                = ${CHROM}"
echo "  DATA_DIR             = ${DATA_DIR}"
echo "  GENOME_PATH          = ${GENOME_PATH}"
echo "  STRIDE               = ${STRIDE}"
echo "  WINDOW_SIZE          = ${WINDOW_SIZE}"
echo "  BATCH_SIZE           = ${BATCH_SIZE}"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"

"${PYTHON}" scripts/run_inference.py \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT}" \
    --chrom "${CHROM}" \
    --data-dir "${DATA_DIR}" \
    --genome-path "${GENOME_PATH}" \
    --stride "${STRIDE}" \
    --window-size "${WINDOW_SIZE}" \
    --batch-size "${BATCH_SIZE}"

echo "[$(date -Iseconds)] Done with exit $?"
