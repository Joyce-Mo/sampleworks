#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l mem_free=32G
#$ -l scratch=10G
#$ -l h_rt=48:00:00
#$ -r y
#$ -m bea
#$ -M joyce.mo@ucsf.edu


set -euo pipefail
date
hostname

## Configuration
SAMPLEWORKS_DIR="/wynton/home/rotation/jqmo/rotation3/sampleworks"
PROTPARDELLE_CONFIG="$HOME/.protpardelle/config.yaml"
PROTPARDELLE_CHECKPOINT="$HOME/.protpardelle/model.pth"
STRUCTURE="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION=1.8
ENSEMBLE_SIZE=4
GUIDANCE_START=130

cd "$SAMPLEWORKS_DIR"

##  Run WITH guidance
echo "Running Protpardelle WITH guidance (CPU)"

pixi run python scripts/protpardelle_pure_guidance.py \
    --config-path "$PROTPARDELLE_CONFIG" \
    --checkpoint-path "$PROTPARDELLE_CHECKPOINT" \
    --structure "$STRUCTURE" \
    --density "$DENSITY" \
    --resolution "$RESOLUTION" \
    --output-dir "${SAMPLEWORKS_DIR}/output/protpardelle_with_guidance" \
    --guidance-start "$GUIDANCE_START" \
    --ensemble-size "$ENSEMBLE_SIZE" \
    --augmentation \
    --align-to-input \
    --device cpu

##  Run WITHOUT guidance (unconditional sampling)
echo "Running Protpardelle WITHOUT guidance (CPU)"

pixi run python scripts/protpardelle_pure_guidance.py \
    --config-path "$PROTPARDELLE_CONFIG" \
    --checkpoint-path "$PROTPARDELLE_CHECKPOINT" \
    --structure "$STRUCTURE" \
    --density "$DENSITY" \
    --resolution "$RESOLUTION" \
    --output-dir "${SAMPLEWORKS_DIR}/output/protpardelle_no_guidance" \
    --ensemble-size "$ENSEMBLE_SIZE" \
    --no-guidance \
    --device cpu

##  End-of-job summary
echo ""
echo "All Protpardelle runs complete."
[[ -n "${JOB_ID:-}" ]] && qstat -j "$JOB_ID"
