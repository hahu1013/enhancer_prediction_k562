#!/bin/bash
#SBATCH --job-name=enh_evaluate
#SBATCH --partition=tier2q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.log
#SBATCH --error=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.err

# Phase 5 — chr10 evaluation. CPU job — no GPU, no --gres. Module loads
# are gcc + miniconda3 only (durable rule; this is a numpy/sklearn/matplotlib
# Python job that needs neither openjdk nor samtools).
#
# Partition note: defaults to zhoulabq+reservation for consistency with the
# rest of the project. PHASE_5_STATUS.md says you can switch to tier3q
# (general CPU queue, no reservation) if the lab queue is busy and tier3q
# is confirmed available — just override --partition / --reservation when
# submitting.
#
# Required env: PREDICTIONS (space-separated name=path pairs) + STRATA.
# Example:
#   PREDICTIONS="from_scratch_best=runs/from_scratch/pred_chr10_best.npz \
#                pretrained_best=runs/pretrained/pred_chr10_best.npz" \
#   STRATA=data/labels/chr10.strata.npy \
#       sbatch scripts/evaluate.sh

set -eo pipefail

module load gcc/12.1.0
module load miniconda3/24.9.2
source activate /gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint

PYTHON=/gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint/bin/python
PROJECT=/gpfs/data/zhou-lab/haoyanghu/projects/enhancer_prediction

# Required — set via env at sbatch time.
PREDICTIONS=${PREDICTIONS:?PREDICTIONS env var is required (space-separated name=path pairs)}
STRATA=${STRATA:?STRATA env var is required (path to chr10.strata.npy)}

# Optional overrides.
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/eval}
CHROM=${CHROM:-chr10}
THRESHOLDS=${THRESHOLDS:-0.1,0.3,0.5,0.7,0.9}
CURVE_POINTS=${CURVE_POINTS:-2000}
SEED=${SEED:-42}

mkdir -p /gpfs/data/zhou-lab/haoyanghu/logs
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT}"

echo "[$(date -Iseconds)] Starting ${SLURM_JOB_NAME:-enh_evaluate} on $(hostname)"
echo "  PREDICTIONS   = ${PREDICTIONS}"
echo "  STRATA        = ${STRATA}"
echo "  OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "  CHROM         = ${CHROM}"
echo "  THRESHOLDS    = ${THRESHOLDS}"
echo "  CURVE_POINTS  = ${CURVE_POINTS}"
echo "  SEED          = ${SEED}"

# $PREDICTIONS is space-separated; word-splitting is intentional here so
# each name=path becomes its own arg to --predictions.
# shellcheck disable=SC2086
"${PYTHON}" scripts/evaluate.py \
    --predictions ${PREDICTIONS} \
    --strata "${STRATA}" \
    --output-dir "${OUTPUT_DIR}" \
    --chrom "${CHROM}" \
    --thresholds "${THRESHOLDS}" \
    --curve-points "${CURVE_POINTS}" \
    --seed "${SEED}"

echo "[$(date -Iseconds)] Done with exit $?"
