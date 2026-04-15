#!/bin/bash
#SBATCH -A bio260134-ai
#SBATCH --job-name=baseline_protpardelle_guidance_04112026
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH -o logs/baseline_protpardelle_guidance_04142026_%j.out
#SBATCH -e logs/baseline_protpardelle_guidance_04142026_%j.err
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=ALL

set -euo pipefail
date
hostname
nvidia-smi

## Load modules
module purge
module load modtree/gpu
module load cuda/12.8.0
module list

## Ensure pixi is in PATH 
export PATH="$HOME/.pixi/bin:$PATH"
echo "Using pixi: $(which pixi) ($(pixi --version))"

## Environment variables
PROTPARDELLE_DIR="/anvil/projects/x-bio260134/jmo/protpardelle-1c"
export PYTHONPATH="${PROTPARDELLE_DIR}/src:${PYTHONPATH:-}"

## Install protpardelle's missing deps into the pixi env
## dm-tree: needed by protpardelle.common.residue_constants
## einops: needed by protpardelle.core.models and protpardelle.core.modules
pixi run pip install dm-tree einops 2>/dev/null || true

## Configuration
SAMPLEWORKS_DIR="/anvil/projects/x-bio260134/jmo/sampleworks"
PROTPARDELLE_CONFIG="/anvil/scratch/x-jmo/protpardelle_runs/cc89_anvil_16139230/cc89_anvil/config.yaml"
PROTPARDELLE_CHECKPOINT="/anvil/scratch/x-jmo/protpardelle_runs/cc89_anvil_16139230/cc89_anvil/checkpoints/epoch44_training_state.pth"
STRUCTURE="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION=1.8
ENSEMBLE_SIZE=4
GUIDANCE_START=100 # 130 to keep this consistent with boltz example, but switch to 100 / 200 for t = 0.5

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
