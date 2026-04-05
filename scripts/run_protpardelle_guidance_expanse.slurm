#!/bin/bash
#SBATCH -A ucb368   
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --mem=92G
#SBATCH --time=1:00:00
#SBATCH -J protpardelle_guidance_4022026_
#SBATCH -o logs/protpardelle_guidance_4022026_%j.out
#SBATCH -e logs/protpardelle_guidance_4022026_%j.err
#SBATCH -p gpu
#SBATCH --mail-user=jqmo@berkeley.edu
#SBATCH --mail-type=all

set -euo pipefail
date
hostname
nvidia-smi

## Load modules
module purge
module load gpu/0.15.4
module load gcc/10.2.0
module load cuda/12.2.1

## Environment variables
export TORCH_CUDA_ARCH_LIST="7.0;8.0"
PROTPARDELLE_DIR="/expanse/lustre/scratch/jmo/temp_project/protpardelle-1c"
export PYTHONPATH="${PROTPARDELLE_DIR}/src:${PYTHONPATH:-}"

## Configuration
SAMPLEWORKS_DIR="/expanse/lustre/scratch/jmo/temp_project/sampleworks"
PROTPARDELLE_CONFIG="${PROTPARDELLE_DIR}/examples/training/cc89.yaml"

## weights
PROTPARDELLE_CHECKPOINT="${PROTPARDELLE_DIR}/model_params/weights/cc89_epoch415.pth"
STRUCTURE="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION=1.8
ENSEMBLE_SIZE=1  # 1 sample per GPU, 4 GPUs = 4 total ensemble members
GUIDANCE_START=130

cd "$SAMPLEWORKS_DIR"

## Run WITH guidance
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

## Run WITHOUT guidance (4 GPUs in parallel)
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
