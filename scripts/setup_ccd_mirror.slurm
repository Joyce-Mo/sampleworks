#!/bin/bash
#SBATCH -A ucb368
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --mem=92G
#SBATCH --time=0:30:00
#SBATCH -J setup_ccd_mirror
#SBATCH -o logs/setup_ccd_mirror_%j.out
#SBATCH -e logs/setup_ccd_mirror_%j.err
#SBATCH -p gpu

set -euo pipefail
date
hostname

SAMPLEWORKS_DIR="/expanse/lustre/scratch/jmo/temp_project/sampleworks"
cd "$SAMPLEWORKS_DIR"

## Download from github readme that takes too long
pixi run -e boltz python -c "from boltz.main import download_boltz2; import pathlib; download_boltz2(pathlib.Path('~/boltz/'))"


echo "=== Step 1: Inspect ccd.pkl ==="
pixi run -e boltz python -c "
import pickle
from pathlib import Path

with open('ccd.pkl', 'rb') as f:
    ccd = pickle.load(f)

print(f'Type: {type(ccd)}')
if isinstance(ccd, dict):
    keys = list(ccd.keys())
    print(f'Number of entries: {len(keys)}')
    print(f'First 10 keys: {keys[:10]}')
    first_key = keys[0]
    print(f'Type of value for key {first_key}: {type(ccd[first_key])}')
    print(f'Value preview: {str(ccd[first_key])[:500]}')
else:
    print(f'Content preview: {str(ccd)[:500]}')
"

echo ""
echo "=== Step 2: Check atomworks CCD mirror utilities ==="
pixi run -e boltz python -c "
import atomworks.io.utils.ccd as ccd_mod
funcs = [x for x in dir(ccd_mod) if 'download' in x.lower() or 'mirror' in x.lower() or 'setup' in x.lower() or 'create' in x.lower()]
print(f'Mirror/download related functions: {funcs}')

# Try to get help on download_ccd_mirror if it exists
if hasattr(ccd_mod, 'download_ccd_mirror'):
    help(ccd_mod.download_ccd_mirror)
elif hasattr(ccd_mod, 'create_ccd_mirror'):
    help(ccd_mod.create_ccd_mirror)
else:
    print('No download/create mirror function found.')
    print(f'All public functions: {[x for x in dir(ccd_mod) if not x.startswith(\"_\")]}')
"

echo ""
echo "=== Step 3: Check what CCD_MIRROR_PATH format is expected ==="
pixi run -e boltz python -c "
import inspect
from atomworks.io.utils.ccd import get_available_ccd_codes_in_mirror
print(inspect.getsource(get_available_ccd_codes_in_mirror))
"

echo ""



echo "=== Done ==="
date
