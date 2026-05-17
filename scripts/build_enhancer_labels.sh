#!/bin/bash
#SBATCH --job-name=build_e_labels
#SBATCH --partition=tier3q
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.log
#SBATCH --error=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.err

# Phase 1 — per-chromosome enhancer-label arrays (Briefing § 4 Phase 1).
# CPU only (~30–60 min serial). No `--gres=gpu` — GPUs on cri22cn416 are not
# SLURM-managed; this job runs on a CPU node.

set -eo pipefail

module load gcc/12.1.0
module load miniconda3/24.9.2

source activate /gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint

PROJECT=/gpfs/data/zhou-lab/haoyanghu/projects/enhancer_prediction
PYTHON=/gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint/bin/python

CHROMS=${CHROMS:-chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/data/labels}
PLS_FLANK=${PLS_FLANK:-500}
BLACKLIST_BED=${BLACKLIST_BED:-/gpfs/data/zhou-lab/haoyanghu/data/annotations/hg38-blacklist.v2.bed.gz}

mkdir -p /gpfs/data/zhou-lab/haoyanghu/logs
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT}"

echo "[$(date -Iseconds)] Starting build_enhancer_labels on $(hostname)"
echo "  CHROMS        = ${CHROMS}"
echo "  OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "  PLS_FLANK     = ${PLS_FLANK}"
echo "  BLACKLIST_BED = ${BLACKLIST_BED}"

"${PYTHON}" scripts/build_enhancer_labels.py \
    --chroms "${CHROMS}" \
    --output-dir "${OUTPUT_DIR}" \
    --pls-flank "${PLS_FLANK}" \
    --blacklist-bed "${BLACKLIST_BED}"

echo "[$(date -Iseconds)] Done with exit $?"
