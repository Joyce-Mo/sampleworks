#!/bin/bash
#SBATCH -A bio260134-ai
#SBATCH --job-name=boltz2_guidance_04092026
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH -o logs/boltz2_guidance_04092026_%j.out
#SBATCH -e logs/boltz2_guidance_04092026_%j.err
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=ALL

set -euo pipefail
date
hostname
nvidia-smi

## Load modules
module purge
module load modtree/gpu
module list

# from sampelworks example: 
# pixi run -e boltz python scripts/boltz2_pure_guidance.py \
#     --model-checkpoint ~/.boltz/boltz2_conf.ckpt \
#     --structure tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif \
#     --density tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4 \
#     --resolution 1.8 \
#     --output-dir output/boltz2_pure_guidance \
#     --guidance-start 130 \
#     --ensemble-size 4 \
#     --augmentation \
#     --align-to-input



## Configuration
SAMPLEWORKS_DIR="/anvil/projects/x-bio260134/jmo/sampleworks"
CHECKPOINT="~/.boltz/boltz2_conf.ckpt"
STRUCTURE="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION=1.8
OUTPUT_DIR="${SAMPLEWORKS_DIR}/output/boltz2_pure_guidance"
GUIDANCE_START=130
ENSEMBLE_SIZE=4

## Run job
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

## End-of-job summary
echo "Job $SLURM_JOB_ID complete."
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State
