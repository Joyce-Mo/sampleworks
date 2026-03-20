#SBATCH --job-name=bolts_guidance_sampleworks_03202026
#SBATCH --output=bolts_guidance_sampleworks_03202026.o%j
#SBATCH --error=bolts_guidance_sampleworks_03202026.e%j
#SBATCH --account=fc_actin
#SBATCH --partition=savio3_gpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --qos=a40_gpu3_normal
#SBATCH --mail-type=all
#SBATCH --mail-user=jqmo@berkeley.edu

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
PROTPARDELLE_CONFIG="$HOME/.protpardelle/config.yaml"
PROTPARDELLE_CHECKPOINT="$HOME/.protpardelle/model.pth"
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
