#!/bin/bash
#SBATCH --job-name=enh_build_strata
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=20:00
#SBATCH --output=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.log
#SBATCH --error=/gpfs/data/zhou-lab/haoyanghu/logs/slurm.%x.%j.err

# Phase 5 § 4.3 — build the cCRE-category side file for stratified AUROC.
# CPU job — no GPU, no --gres. ~20 s wall time on chr10.
# Module loads: gcc + miniconda3 only (durable rule).

set -eo pipefail

module load gcc/12.1.0
module load miniconda3/24.9.2
source activate /gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint

PYTHON=/gpfs/data/zhou-lab/haoyanghu/software/conda/envs/af_constraint/bin/python
PROJECT=/gpfs/data/zhou-lab/haoyanghu/projects/enhancer_prediction

CHROM=${CHROM:-chr10}
CCRE_BED=${CCRE_BED:-/gpfs/data/zhou-lab/haoyanghu/data/annotations/GRCh38-cCREs.bed}
CHROM_SIZES=${CHROM_SIZES:-/gpfs/data/zhou-lab/haoyanghu/data/reference/hg38.chrom.sizes}
OUTPUT=${OUTPUT:-${PROJECT}/data/labels/${CHROM}.strata.npy}
OUTPUT_JSON=${OUTPUT_JSON:-${PROJECT}/data/labels/${CHROM}.strata.build.json}

mkdir -p /gpfs/data/zhou-lab/haoyanghu/logs
mkdir -p "$(dirname "${OUTPUT}")"
cd "${PROJECT}"

echo "[$(date -Iseconds)] Starting ${SLURM_JOB_NAME:-enh_build_strata} on $(hostname)"
echo "  CHROM        = ${CHROM}"
echo "  CCRE_BED     = ${CCRE_BED}"
echo "  CHROM_SIZES  = ${CHROM_SIZES}"
echo "  OUTPUT       = ${OUTPUT}"
echo "  OUTPUT_JSON  = ${OUTPUT_JSON}"

"${PYTHON}" scripts/build_strata.py \
    --chrom "${CHROM}" \
    --ccre-bed "${CCRE_BED}" \
    --chrom-sizes "${CHROM_SIZES}" \
    --output "${OUTPUT}" \
    --output-json "${OUTPUT_JSON}"

echo "[$(date -Iseconds)] Done with exit $?"
