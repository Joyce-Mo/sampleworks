#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l mem_free=16G
#$ -l scratch=10G
#$ -l h_rt=24:00:00
#$ -l gpu=1
#$ -r y
#$ -m bea
#$ -M joyce.mo@ucsf.edu

# ==============================================================================
# Run Boltz2 pure guidance with real-space density reward on Wynton HPC
#
# ── FIRST-TIME SETUP (run on the LOGIN NODE, compute nodes have no internet) ──
#
#   Step 1: Install pixi
#     curl -fsSL https://pixi.sh/install.sh | bash
#     source ~/.bashrc
#
#   Step 2: Clone sampleworks and install all environments
#     cd /wynton/home/rotation/jqmo
#     git clone <sampleworks-repo-url> sampleworks
#     cd sampleworks
#     pixi install -a            # builds all envs (boltz, protenix, rf3, etc.)
#
#   Step 3: Download the Boltz2 model checkpoint (~4 GB)
#     pixi run -e boltz python -c \
#       "from boltz.main import download_boltz2; import pathlib; \
#        download_boltz2(pathlib.Path.home() / '.boltz')"
#     # This creates ~/.boltz/boltz2_conf.ckpt
#
#   Step 4: Submit this job
#     cd /wynton/home/rotation/jqmo/sampleworks
#     qsub scripts/run_boltz2_guidance.sh
#
#   Step 5: Monitor
#     qstat -u $(whoami)          # check job status
#     cat output/boltz2_pure_guidance/run.log  # check logs after completion
# ==============================================================================

set -euo pipefail
date
hostname

## ── Configuration (edit these paths for your setup) ────────────────────────────
SAMPLEWORKS_DIR="/wynton/home/rotation/jqmo/sampleworks"
CHECKPOINT="$HOME/.boltz/boltz2_conf.ckpt"
STRUCTURE="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION=1.8
OUTPUT_DIR="${SAMPLEWORKS_DIR}/output/boltz2_pure_guidance"
GUIDANCE_START=130
ENSEMBLE_SIZE=4

## ── Run ────────────────────────────────────────────────────────────────────────
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
