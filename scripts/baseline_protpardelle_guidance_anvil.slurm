#!/bin/bash
#SBATCH -A bio260134-ai
#SBATCH --job-name=protpardelle_guidance_04272026
#SBATCH -p ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH -o logs/protpardelle_guidance_04272026_%j.out
#SBATCH -e logs/protpardelle_guidance_04272026_%j.err
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

## Protpardelle source is added via PYTHONPATH (not a pixi feature).
## Its deps (dm-tree, einops) are declared in pyproject.toml so pixi resolves them.
## See: https://github.com/ProteinDesignLab/protpardelle
PROTPARDELLE_DIR="/anvil/projects/x-bio260134/jmo/protpardelle-1c"
export PYTHONPATH="${PROTPARDELLE_DIR}/src:${PYTHONPATH:-}"

## Shared configuration
SAMPLEWORKS_DIR="/anvil/projects/x-bio260134/jmo/sampleworks"
PROTPARDELLE_CONFIG="/anvil/projects/x-bio260134/jmo/protpardelle-1c/examples/training/cc89.yaml"
PROTPARDELLE_CHECKPOINT="/anvil/projects/x-bio260134/jmo/protpardelle-1c/model_params/weights/cc89_epoch415.pth"
ENSEMBLE_SIZE=4
GUIDANCE_START=250  # t=0.5 with 500 steps

## Protein definitions for test
## 6B8X (PTP1B, 298 residues)
STRUCT_6B8X="${SAMPLEWORKS_DIR}/tests/resources/Joyce_tests/6B8X_single_001_density_input.cif"
## PDB_6B8X="${SAMPLEWORKS_DIR}/tests/resources/Joyce_tests/6B8X_single_001_density_input.pdb"
DENSITY_6B8X="${SAMPLEWORKS_DIR}/tests/resources/Joyce_tests/6B8X_uniform_1.00A.ccp4"
RESOLUTION_6B8X=1.8
NRES_6B8X=298

## 1VME (402 residues)
STRUCT_1VME="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY_1VME="${SAMPLEWORKS_DIR}/tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
RESOLUTION_1VME=1.8

## Output base
OUT="${SAMPLEWORKS_DIR}/output/protpardelle"

cd "$SAMPLEWORKS_DIR"

## Helper: run sampleworks protpardelle script
## Usage: run_sw <output_dir> <structure> <density> <resolution> [extra_args...]
run_sw() {
    local out_dir="$1" struct="$2" density="$3" resolution="$4"
    shift 4
    pixi run python scripts/protpardelle_pure_guidance.py \
        --config-path "$PROTPARDELLE_CONFIG" \
        --checkpoint-path "$PROTPARDELLE_CHECKPOINT" \
        --structure "$struct" \
        --density "$density" \
        --resolution "$resolution" \
        --ensemble-size "$ENSEMBLE_SIZE" \
        --t-start 0.5 \
        --align-to-input \
        --output-dir "$out_dir" \
        "$@"
}

echo "  6B8X (PTP1B)"

## 6B8X Run 0: ppd_helper.py baseline (sanity check)
## Runs protpardelle's own sampling directly to establish ground truth RMSD.
echo "[6B8X] Run 0: ppd_helper.py baseline"
mkdir -p "${OUT}/6b8x_ppd_helper_BASeLINE"

pixi run python -c "
import torch, sys
sys.path.insert(0, '${PROTPARDELLE_DIR}/representation_extraction')
from ppd_helper import ProtpardelleHelper

helper = ProtpardelleHelper(max_steps=500, device='cuda', model_name='cc89', epoch='415')
coords = helper.initialize_partial_noise('${PDB_6B8X}', [${NRES_6B8X}], t=0.5, use_seq=True)

prev_pred = None
tspan_list = [(1.0 - i/500, 1.0 - (i+1)/500) for i in range(250, 500)]
for tspan in tspan_list:
    coords, xhat = helper.stepper(coords, tspan, prev_pred=prev_pred, step_scale=1.0, align=True)
    prev_pred = helper.struct_self_cond_out

helper.save_PDBs(coords, '${OUT}/6b8x_ppd_helper_baseline', '6b8x', 0)
print('6B8X ppd_helper baseline saved.')
"

## 6B8X Run 1: No guidance, ODE
echo "[6B8X] Run 1: No guidance, ODE"
run_sw "${OUT}/6b8x_no_guidance_ode" "$STRUCT_6B8X" "$DENSITY_6B8X" "$RESOLUTION_6B8X" \
    --no-guidance

## 6B8X Run 2: No guidance, Langevin
echo "[6B8X] Run 2: No guidance, Langevin"
run_sw "${OUT}/6b8x_no_guidance_langevin" "$STRUCT_6B8X" "$DENSITY_6B8X" "$RESOLUTION_6B8X" \
    --no-guidance --langevin

## 6B8X Run 3: With guidance, ODE
echo "[6B8X] Run 3: With guidance, ODE"
run_sw "${OUT}/6b8x_guidance_ode" "$STRUCT_6B8X" "$DENSITY_6B8X" "$RESOLUTION_6B8X" \
    --guidance-start "$GUIDANCE_START"

## 6B8X Run 4: With guidance, Langevin
echo "[6B8X] Run 4: With guidance, Langevin"
run_sw "${OUT}/6b8x_guidance_langevin" "$STRUCT_6B8X" "$DENSITY_6B8X" "$RESOLUTION_6B8X" \
    --guidance-start "$GUIDANCE_START" --langevin

echo ""
echo "  1VME"

## 1VME Run 1: No guidance, ODE
echo "[1VME] Run 1: No guidance, ODE"
run_sw "${OUT}/1vme_no_guidance_ode" "$STRUCT_1VME" "$DENSITY_1VME" "$RESOLUTION_1VME" \
    --no-guidance

## 1VME Run 2: No guidance, Langevin
echo "[1VME] Run 2: No guidance, Langevin"
run_sw "${OUT}/1vme_no_guidance_langevin" "$STRUCT_1VME" "$DENSITY_1VME" "$RESOLUTION_1VME" \
    --no-guidance --langevin

## 1VME Run 3: With guidance, ODE
echo "[1VME] Run 3: With guidance, ODE"
run_sw "${OUT}/1vme_guidance_ode" "$STRUCT_1VME" "$DENSITY_1VME" "$RESOLUTION_1VME" \
    --guidance-start "$GUIDANCE_START"

## 1VME Run 4: With guidance, Langevin
echo "[1VME] Run 4: With guidance, Langevin"
run_sw "${OUT}/1vme_guidance_langevin" "$STRUCT_1VME" "$DENSITY_1VME" "$RESOLUTION_1VME" \
    --guidance-start "$GUIDANCE_START" --langevin

## End-of-job summary
echo ""
echo "All runs complete."
echo "Job $SLURM_JOB_ID finished at $(date)"
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State
