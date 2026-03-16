#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -l mem_free=16G
#$ -l scratch=10G
#$ -l h_rt=24:00:00
#$ -q gpu.q
#$ -l gpu_mem=16G
#$ -r y
#$ -m bea
#$ -M joyce.mo@ucsf.edu

# ==============================================================================
# Run Protpardelle with and without guidance on Wynton HPC
#
# ── FIRST-TIME SETUP (run on the LOGIN NODE, compute nodes have no internet) ──
#
#   Step 1: Install pixi
#     curl -fsSL https://pixi.sh/install.sh | bash
#     source ~/.bashrc
#
#   Step 2: Clone sampleworks and install environments
#     cd /wynton/home/rotation/jqmo
#     git clone <sampleworks-repo-url> sampleworks
#     cd sampleworks
#     pixi install -a
#
#   Step 3: Install protpardelle into the default pixi environment
#     Protpardelle is NOT in the pixi environment system yet (unlike boltz/
#     protenix/rf3), so it must be installed manually.
#
#     Option A — Install into the default pixi env:
#       cd /wynton/home/rotation/jqmo
#       git clone <protpardelle-repo-url> protpardelle
#       cd sampleworks
#       pixi shell                   # activates the default env
#       pip install -e ../protpardelle
#       exit                         # leave pixi shell
#
#     Option B — Use a standalone conda env (if protpardelle has conflicting deps):
#       module load CBI miniforge3
#       conda create -y -n protpardelle python=3.11
#       conda activate protpardelle
#       cd /wynton/home/rotation/jqmo
#       git clone <protpardelle-repo-url> protpardelle
#       pip install -e protpardelle
#       cd sampleworks
#       pip install -e .             # install sampleworks into same env
#       conda deactivate
#       # If using Option B, uncomment the conda activation lines below and
#       # replace "pixi run python" with just "python" in the run commands.
#
#   Step 4: Obtain the protpardelle checkpoint and config
#     mkdir -p ~/.protpardelle
#     cp /path/to/protpardelle_config.yaml ~/.protpardelle/config.yaml
#     cp /path/to/protpardelle_checkpoint.pth ~/.protpardelle/model.pth
#
#   Step 5: Submit this job
#     cd /wynton/home/rotation/jqmo/sampleworks
#     qsub scripts/run_protpardelle_guidance.sh
#
#   Step 6: Monitor
#     qstat -u $(whoami)
#     cat output/protpardelle_with_guidance/run.log
#     cat output/protpardelle_no_guidance/run.log
# ==============================================================================

set -euo pipefail
date
hostname

## ── Configuration (edit these paths for your setup) ────────────────────────────
SAMPLEWORKS_DIR="/wynton/home/rotation/jqmo/sampleworks"
PROTPARDELLE_CONFIG="$HOME/.protpardelle/config.yaml"
PROTPARDELLE_CHECKPOINT="$HOME/.protpardelle/model.pth"
STRUCTURE="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION=1.8
ENSEMBLE_SIZE=4
GUIDANCE_START=130

## ── Activate environment ───────────────────────────────────────────────────────
# Uncomment the following two lines if using Option B (conda env) from setup:
#   module load CBI miniforge3
#   source activate protpardelle

cd "$SAMPLEWORKS_DIR"

## ── Run WITH guidance ──────────────────────────────────────────────────────────
echo "============================================"
echo "Running Protpardelle WITH guidance"
echo "============================================"

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
    --align-to-input

## ── Run WITHOUT guidance (unconditional sampling) ──────────────────────────────
echo ""
echo "============================================"
echo "Running Protpardelle WITHOUT guidance"
echo "============================================"

pixi run python scripts/protpardelle_pure_guidance.py \
    --config-path "$PROTPARDELLE_CONFIG" \
    --checkpoint-path "$PROTPARDELLE_CHECKPOINT" \
    --structure "$STRUCTURE" \
    --density "$DENSITY" \
    --resolution "$RESOLUTION" \
    --output-dir "${SAMPLEWORKS_DIR}/output/protpardelle_no_guidance" \
    --ensemble-size "$ENSEMBLE_SIZE" \
    --no-guidance

## ── End-of-job summary ────────────────────────────────────────────────────────
echo ""
echo "All Protpardelle runs complete."
[[ -n "${JOB_ID:-}" ]] && qstat -j "$JOB_ID"
