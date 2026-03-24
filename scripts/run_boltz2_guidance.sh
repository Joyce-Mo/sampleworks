#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y ## merges stdout and stderr 
#$ -l mem_free=16G 
#$ -l scratch=10G
#$ -l h_rt=48:00:00
#$ -q gpu.q
#$ -l gpu_mem=16G
#$ -r y
#$ -m bea
#$ -M joyce.mo@ucsf.edu

set -euo pipefail
date
hostname

## Configuration (based on example lol)
SAMPLEWORKS_DIR="/wynton/home/rotation/jqmo/rotation3/sampleworks"
CHECKPOINT="$HOME/.boltz/boltz2_conf.ckpt"
STRUCTURE="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION=1.8
OUTPUT_DIR="${SAMPLEWORKS_DIR}/output/boltz2_pure_guidance"
GUIDANCE_START=130
ENSEMBLE_SIZE=4

##  Run job 
cd "$SAMPLEWORKS_DIR"

pixi run -e boltz python scripts/boltz2_pure_guidance.py \
    --model-checkpoint "$CHECKPOINT" \
    --structure "$STRUCTURE" \
    --density "$DENSITY" \
    --resolution "$RESOLUTION" \
    --output-dir "$OUTPUT_DIR" \
    --guidance-start "$GUIDANCE_START" \
    --ensemble-size "$ENSEMBLE_SIZE" \
    --augmentation \
    --align-to-input

## ── End-of-job summary ────────────────────────────────────────────────────────
[[ -n "${JOB_ID:-}" ]] && qstat -j "$JOB_ID"
