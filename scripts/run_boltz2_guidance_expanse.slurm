#!/bin/bash
#SBATCH -A ucb368   
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH -J boltz2_guidance_4052026_
#SBATCH -o logs/boltz2_guidance_4052026_%j.out
#SBATCH -e logs/boltz2_guidance_4052026_%j.err
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
module load cuda/11.0.2

## Environment variables
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_CUDA_ARCH_LIST="7.0;8.0"

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

## Ensure the Boltz2 mols/ directory exists next to the checkpoint.
## The Boltz wrapper derives its cache dir from Path(checkpoint).parent and
## loads per-component molecule pickles from "<cache>/mols/". Without it
## load_molecules() raises "CCD component ALA not found".
## boltz.main.download_boltz2() times out on this cluster, so fetch just
## the mols tarball directly from HuggingFace.
CHECKPOINT_DIR="$(dirname "$CHECKPOINT")"
MOLS_DIR="${CHECKPOINT_DIR}/mols"
if [ ! -d "$MOLS_DIR" ] || [ -z "$(ls -A "$MOLS_DIR" 2>/dev/null || true)" ]; then
    echo "mols/ not found at $MOLS_DIR — downloading mols.tar from HuggingFace"
    curl -L --retry 5 --retry-delay 10 \
        -o "${CHECKPOINT_DIR}/mols.tar" \
        "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"
    tar -xf "${CHECKPOINT_DIR}/mols.tar" -C "$CHECKPOINT_DIR"
    rm -f "${CHECKPOINT_DIR}/mols.tar"
    echo "mols/ ready at $MOLS_DIR"
fi

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
## sacct lives on login nodes only on Expanse; don't fail the job if missing.
if command -v sacct >/dev/null 2>&1; then
    sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State || true
fi
