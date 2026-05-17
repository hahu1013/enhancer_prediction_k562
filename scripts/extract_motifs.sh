#!/bin/bash
#SBATCH --job-name=enh_extract_motifs
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=30:00
#SBATCH --output=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.log
#SBATCH --error=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.err

# Phase 6.2 — cluster saliency profiles + build per-cluster PWMs + save
# logos. CPU job (numpy + sklearn + matplotlib). No GPU. No --gres.
# Module loads: gcc + miniconda3 only (durable rule).

set -eo pipefail

module load gcc/12.1.0
module load miniconda3/24.9.2
source activate /gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint

PYTHON=/gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint/bin/python
PROJECT=/gpfs/data/zhou-lab/haoyanghu/projects/enhancer_prediction

ISM_OUTPUT=${ISM_OUTPUT:-${PROJECT}/interpretation/ism_saliency_chr10.npz}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/interpretation}
N_CLUSTERS=${N_CLUSTERS:-10}
MIN_CLUSTER_SIZE=${MIN_CLUSTER_SIZE:-20}
LINKAGE=${LINKAGE:-average}
SEED=${SEED:-42}

mkdir -p /gpfs/data/zhou-lab/haoyanghu/logs
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT}"

echo "[$(date -Iseconds)] Starting ${SLURM_JOB_NAME:-enh_extract_motifs} on $(hostname)"
echo "  ISM_OUTPUT       = ${ISM_OUTPUT}"
echo "  OUTPUT_DIR       = ${OUTPUT_DIR}"
echo "  N_CLUSTERS       = ${N_CLUSTERS}"
echo "  MIN_CLUSTER_SIZE = ${MIN_CLUSTER_SIZE}"
echo "  LINKAGE          = ${LINKAGE}"

"${PYTHON}" scripts/extract_motifs.py \
    --ism-output "${ISM_OUTPUT}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-clusters "${N_CLUSTERS}" \
    --min-cluster-size "${MIN_CLUSTER_SIZE}" \
    --linkage "${LINKAGE}" \
    --seed "${SEED}"

echo "[$(date -Iseconds)] Done with exit $?"
