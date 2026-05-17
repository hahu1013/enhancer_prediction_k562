#!/bin/bash
#SBATCH --job-name=enh_match_jaspar
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=10:00
#SBATCH --output=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.log
#SBATCH --error=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.err

# Phase 6.3 — Tomtom-style match of cluster PWMs against JASPAR 2024 CORE
# vertebrates non-redundant PFMs. CPU only. <1 minute. Module loads: gcc
# + miniconda3 only (durable rule).
#
# Prerequisite: the JASPAR PFM file must exist at JASPAR. See
# docs/PHASE_6_STATUS.md for the download URL.

set -eo pipefail

module load gcc/12.1.0
module load miniconda3/24.9.2
source activate /gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint

PYTHON=/gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint/bin/python
PROJECT=/gpfs/data/zhou-lab/haoyanghu/projects/enhancer_prediction

CLUSTER_PWMS=${CLUSTER_PWMS:-${PROJECT}/interpretation/cluster_pwms.npz}
JASPAR=${JASPAR:-/gpfs/data/zhou-lab/haoyanghu/data/jaspar/JASPAR2024_CORE_vertebrates_nr.txt}
OUTPUT=${OUTPUT:-${PROJECT}/interpretation/jaspar_matches.json}
TOP_N_MATCHES=${TOP_N_MATCHES:-3}
MAX_OFFSET=${MAX_OFFSET:-5}

mkdir -p /gpfs/data/zhou-lab/haoyanghu/logs
mkdir -p "$(dirname "${OUTPUT}")"
cd "${PROJECT}"

echo "[$(date -Iseconds)] Starting ${SLURM_JOB_NAME:-enh_match_jaspar} on $(hostname)"
echo "  CLUSTER_PWMS    = ${CLUSTER_PWMS}"
echo "  JASPAR          = ${JASPAR}"
echo "  OUTPUT          = ${OUTPUT}"
echo "  TOP_N_MATCHES   = ${TOP_N_MATCHES}"
echo "  MAX_OFFSET      = ${MAX_OFFSET}"

"${PYTHON}" scripts/match_jaspar.py \
    --cluster-pwms "${CLUSTER_PWMS}" \
    --jaspar-pfm-file "${JASPAR}" \
    --top-n-matches "${TOP_N_MATCHES}" \
    --max-offset "${MAX_OFFSET}" \
    --output "${OUTPUT}"

echo "[$(date -Iseconds)] Done with exit $?"
