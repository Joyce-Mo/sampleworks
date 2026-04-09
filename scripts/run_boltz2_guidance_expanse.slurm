#!/bin/bash
#SBATCH -A ucb368   
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus=a100:4
#SBATCH --mem=128G
#SBATCH --time=00:10:00
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
## NOTE: PyTorch in the pixi `boltz` env is built against CUDA 12.6 (py312_cu126),
## so the system cuda/11.0.2 + gcc/7.2.0 combo we used previously caused the
## dilate_points_cuda JIT compile to fail mid-run with "CUDA is not available".
## The pixi env declares `cuda-toolkit >=12,<13`, so nvcc is provided by the
## env itself — we just need a host compiler new enough for CUDA 12 and we
## must NOT load an older system CUDA that shadows the env's nvcc/libs.
module purge
module load gpu/0.15.4
module load gcc/10.2.0
module load cuda/12.2.1

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

## Point torch.utils.cpp_extension at the pixi-provided CUDA toolkit so the
## dilate_points_cuda JIT build uses the same CUDA the boltz env's torch was
## compiled against (cu126). Without this, torch falls back to whatever nvcc
## is on PATH and the resulting .so fails to load (or never builds), which
## surfaces later as `RuntimeError: CUDA is not available` from the custom op.
PIXI_BOLTZ_PREFIX="${SAMPLEWORKS_DIR}/.pixi/envs/boltz"
export CUDA_HOME="$PIXI_BOLTZ_PREFIX"
export PATH="${PIXI_BOLTZ_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${PIXI_BOLTZ_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
## A single arch is enough for the a100 partition and avoids over-long JIT builds.
export TORCH_CUDA_ARCH_LIST="8.0"
## Give torch a writable kernel cache (the default ~/.cache/torch/kernels lives
## on a read-only path on this node) and a per-job extensions dir so concurrent
## GPU workers don't race on the same JIT build directory.
export TORCH_EXTENSIONS_DIR="${SAMPLEWORKS_DIR}/.torch_extensions/${SLURM_JOB_ID:-local}"
export PYTORCH_KERNEL_CACHE_PATH="${SAMPLEWORKS_DIR}/.torch_kernel_cache"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$PYTORCH_KERNEL_CACHE_PATH"

## Pre-build the dilate_points_cuda extension once, serially, before fanning
## out across GPUs. Otherwise the four background workers all race to JIT
## compile into the same directory and one of them wins while the others see
## a half-written .so and crash with the import error we hit previously.
echo "Pre-building dilate_points_cuda extension..."
pixi run -e boltz python -c "from sampleworks.core.forward_models.xray.real_space_density_deps.ops.csrc import dilate_points_cuda; print('dilate_points_cuda ready')"

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
