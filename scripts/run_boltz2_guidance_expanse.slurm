#!/bin/bash
#SBATCH -A ucb368   
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --mem=92G
#SBATCH --time=1:00:00
#SBATCH -J boltz2_guidance_4012026_
#SBATCH -o logs/boltz2_guidance_4012026_$SLURM_JOB_ID.out
#SBATCH -e logs/boltz2_guidance_4012026_$SLURM_JOB_ID.err
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
module load gcc/7.2.0
module load cuda
module list cuda/11.0.2

## Environment varialbes 
export CCD_MIRROR_PATH=/expanse/lustre/scratch/jmo/temp_project/sampleworks/ccd.pkl
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}         

## Configuration
SAMPLEWORKS_DIR="/expanse/lustre/scratch/jmo/temp_project/sampleworks"
CHECKPOINT="${SAMPLEWORKS_DIR}/boltz2_conf.ckpt"
STRUCTURE="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION=1.8
OUTPUT_DIR="${SAMPLEWORKS_DIR}/output/boltz2_pure_guidance"
GUIDANCE_START=130
ENSEMBLE_SIZE=1  # 1 sample per GPU, 4 GPUs = 4 total ensemble members

cd "$SAMPLEWORKS_DIR"

## Launch 1 ensemble member per GPU in parallel
for GPU_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    pixi run -e boltz python scripts/boltz2_pure_guidance.py \
        --model-checkpoint "$CHECKPOINT" \
        --structure "$STRUCTURE" \
        --density "$DENSITY" \
        --resolution "$RESOLUTION" \
        --output-dir "${OUTPUT_DIR}/gpu${GPU_ID}" \
        --guidance-start "$GUIDANCE_START" \
        --ensemble-size "$ENSEMBLE_SIZE" \
        --augmentation \
        --align-to-input &
done

## Wait for all background jobs to finish
wait
echo "All 4 ensemble members complete."

## End-of-job summary
echo "Job $SLURM_JOB_ID complete."
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State
