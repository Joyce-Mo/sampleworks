#!/bin/bash
#SBATCH -A ucb368      
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH -J baseline_protpardelle_guidance_%m%d%Y
#SBATCH -o baseline_protpardelle_guidance_%m%d%Y_%j.out
#SBATCH -e baseline_protpardelle_guidance_%m%d%Y_%j.err
#SBATCH -p gpu
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=all

set -euo pipefail
date
hostname
nvidia-smi

## Load modules
module purge
module load modtree/gpu
module list

## Configuration
SAMPLEWORKS_DIR="$HOME/sampleworks"
PROTPARDELLE_CONFIG="/anvil/scratch/x-jmo/protpardelle_runs/cc89_anvil_16139230/cc89_anvil/config.yaml"
PROTPARDELLE_CHECKPOINT="/anvil/scratch/x-jmo/protpardelle_runs/cc89_anvil_16139230/cc89_anvil/checkpoints/epoch44_training_state.pth"
STRUCTURE="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION=1.8
ENSEMBLE_SIZE=4
GUIDANCE_START=130

## Run job
cd "$SAMPLEWORKS_DIR"

## Run WITH guidance
echo "Running Protpardelle WITH guidance"

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

## Run WITHOUT guidance (unconditional sampling)
echo "Running Protpardelle WITHOUT guidance"

pixi run python scripts/protpardelle_pure_guidance.py \
    --config-path "$PROTPARDELLE_CONFIG" \
    --checkpoint-path "$PROTPARDELLE_CHECKPOINT" \
    --structure "$STRUCTURE" \
    --density "$DENSITY" \
    --resolution "$RESOLUTION" \
    --output-dir "${SAMPLEWORKS_DIR}/output/protpardelle_no_guidance" \
    --ensemble-size "$ENSEMBLE_SIZE" \
    --no-guidance

## End-of-job summary
echo ""
echo "All Protpardelle runs complete."
echo "Job $SLURM_JOB_ID complete."
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State
