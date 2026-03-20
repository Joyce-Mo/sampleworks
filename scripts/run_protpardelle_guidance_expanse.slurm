#!/bin/bash
#SBATCH -A myallocation       # Allocation name (replace with your allocation)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --mem=92G
#SBATCH --time=24:00:00
#SBATCH -J protpardelle_guid
#SBATCH -o protpardelle_guidance.o%j
#SBATCH -e protpardelle_guidance.e%j
#SBATCH -p gpu
#SBATCH --mail-user=joyce.mo@ucsf.edu
#SBATCH --mail-type=all

set -euo pipefail
date
hostname
nvidia-smi

## Load modules
module purge
module load gpu
module list

## Configuration
SAMPLEWORKS_DIR="$HOME/sampleworks"
PROTPARDELLE_CONFIG="$HOME/.protpardelle/config.yaml"
PROTPARDELLE_CHECKPOINT="$HOME/.protpardelle/model.pth"
STRUCTURE="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION=1.8
ENSEMBLE_SIZE=1  # 1 sample per GPU, 4 GPUs = 4 total ensemble members
GUIDANCE_START=130

cd "$SAMPLEWORKS_DIR"

## ── Run WITH guidance (4 GPUs in parallel) ──────────────────────────────────
echo "Running Protpardelle WITH guidance (4 GPUs)"

for GPU_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    pixi run python scripts/protpardelle_pure_guidance.py \
        --config-path "$PROTPARDELLE_CONFIG" \
        --checkpoint-path "$PROTPARDELLE_CHECKPOINT" \
        --structure "$STRUCTURE" \
        --density "$DENSITY" \
        --resolution "$RESOLUTION" \
        --output-dir "${SAMPLEWORKS_DIR}/output/protpardelle_with_guidance/gpu${GPU_ID}" \
        --guidance-start "$GUIDANCE_START" \
        --ensemble-size "$ENSEMBLE_SIZE" \
        --augmentation \
        --align-to-input &
done

wait
echo "Guided runs complete."

## ── Run WITHOUT guidance (4 GPUs in parallel) ───────────────────────────────
echo "Running Protpardelle WITHOUT guidance (4 GPUs)"

for GPU_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    pixi run python scripts/protpardelle_pure_guidance.py \
        --config-path "$PROTPARDELLE_CONFIG" \
        --checkpoint-path "$PROTPARDELLE_CHECKPOINT" \
        --structure "$STRUCTURE" \
        --density "$DENSITY" \
        --resolution "$RESOLUTION" \
        --output-dir "${SAMPLEWORKS_DIR}/output/protpardelle_no_guidance/gpu${GPU_ID}" \
        --ensemble-size "$ENSEMBLE_SIZE" \
        --no-guidance &
done

wait
echo "Unconditional runs complete."

## End-of-job summary
echo ""
echo "All Protpardelle runs complete."
echo "Job $SLURM_JOB_ID complete."
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State
