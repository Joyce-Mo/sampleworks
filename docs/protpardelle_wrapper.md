# Protpardelle Wrapper for Sampleworks

This document describes the integration of [Protpardelle](https://github.com/ProteinDesignLab/protpardelle) into sampleworks for density-guided protein structure sampling.

## Overview

Protpardelle is an EDM-style (Karras et al., 2022) unconditional protein diffusion model that operates in residue-atom37 coordinate space `[B, L, 37, 3]`. Sampleworks expects flat atom coordinates `[B, n_atoms, 3]`. The wrapper translates between these representations using differentiable scatter/gather operations so that gradient-based guidance (DPS) can flow end-to-end.

The implementation is based on `ppd_helper.py` and `ppd_helper_dru_langevin.py` from the protpardelle-1c repository, written by Dru. Those scripts are the reference for correct sampling behavior with the cc89 checkpoint.

## Architecture

```
SLURM script
    |
    v
protpardelle_pure_guidance.py      (entry point, sampling loop, Langevin noise)
    |
    +-- ProtpardelleWrapper        (model interface: featurize, step, initialize_from_prior)
    +-- AF3EDMSampler              (noise schedule, Euler step, Kabsch alignment)
    +-- PureGuidance / inline loop (orchestration)
    +-- RealSpaceRewardFunction    (density guidance, only when --no-guidance is NOT set)
```

## Key Files

| File | Role |
|------|------|
| `src/sampleworks/models/protpardelle/wrapper.py` | `FlowModelWrapper` protocol implementation |
| `scripts/protpardelle_pure_guidance.py` | Entry-point script for sampling and guidance |
| `scripts/baseline_protpardelle_guidance_anvil.slurm` | SLURM batch script for Anvil HPC |
| `src/sampleworks/core/samplers/edm.py` | AF3EDMSampler with protpardelle-compatible config |

## Wrapper Implementation Details

### Coordinate Representation Translation

Protpardelle uses `[B, L, 37, 3]` (residue x atom37). Sampleworks uses `[B, n_atoms, 3]` (flat, real atoms only). The wrapper maintains a mapping between these via `real_atom_indices`, the flat indices within `[L*37]` where real (non-ghost) atoms live.

- **Flat to atom37** (in `step()`): `torch.scatter` places flat coords into `[B, L*37, 3]`, then reshape to `[B, L, 37, 3]`.
- **Atom37 to flat** (in `step()`): reshape to `[B, L*37, 3]`, then `torch.gather` extracts real atoms.

Both operations are differentiable, preserving gradient flow for guidance.

### Ghost Atom Handling

Ghost atoms (atom37 positions where `atom_mask == 0`) must be explicitly zeroed before the model forward pass. This matches ppd_helper.py:

```python
# ppd_helper.py
mask37 = atom37_mask_from_aatype(self.seq_final, seq_mask).bool()
xt[~mask37] = 0

# wrapper.py
ghost_mask = ~cond.atom_mask.bool()
noisy_coords[ghost_mask] = 0.0
```

### Center of Mass

The starting structure (`x_init`) is centered at the origin in `featurize()`. The center of mass is computed from real atoms only (ghost atoms are excluded since `x_init` is gathered via `real_atom_indices`). The sampler does NOT re-center at each step (`center_each_step=False`), matching ppd_helper.py.

### Residue Index and Chain Index

ppd_helper.py uses sequential 1-indexed residue indices (`torch.arange(1, Nres+1)`) and all-zeros chain index. This matches how protpardelle was trained (single-chain proteins with sequential numbering). The wrapper does the same, ignoring raw PDB `res_id` values which can start at arbitrary numbers or have gaps.

### Self-Conditioning

Protpardelle uses structural self-conditioning: the model's predicted clean structure from the previous step is divided by `sigma_data` and fed back as `struct_self_cond` at the next step. The wrapper stores the full batch of self-conditioning (each ensemble member gets its own), matching ppd_helper.py behavior.

### Noise Schedule

Protpardelle's noise schedule uses the same Karras formula as AF3 but with different parameters and opposite time direction:

| Parameter | AF3 default | Protpardelle (cc89) |
|-----------|-------------|---------------------|
| sigma_data | 16.0 | 10.0 (hardcoded in `diffusion.noise_schedule` default) |
| s_max | 160.0 | 80.0 |
| s_min | 4e-4 | 0.001 |
| rho (p) | 7.0 | 7.0 |

**Important**: The model's internal `sigma_data` (used for EDM preconditioning: c_in, c_skip, c_out) may differ from the noise schedule's `sigma_data` (always 10.0). The config `auto_calc_sigma_data: True` computes the model's value from training data. The script derives `sigma_data_for_schedule` from the model's own noise schedule function to guarantee an exact match with ppd_helper.py.

### sigma_max Initialization

When starting from pure noise, the initial coordinates are scaled by `sigma_max` from the model's noise schedule:

```python
# ppd_helper.py
coords = torch.randn(...)
coords *= self.noise_schedule(1.0)

# wrapper.py initialize_from_prior()
sigma_max = self.model.sampling_noise_schedule_default(torch.tensor(1.0))
return torch.randn(...) * sigma_max
```

This is critical for the pure ODE sampler (`gamma_0=0`) because there is no stochastic noise injection at the first step to rescue an unscaled initialization.

### Model Forward Call

The wrapper passes `run_mpnn_model=False` to match ppd_helper.py behavior for cc89 (`predict_seq=False`). The model's MiniMPNN does not modify denoised coordinates but produces sequence self-conditioning that would diverge from the reference pipeline if enabled.

## Sampling Modes

### Partial Diffusion (recommended, `--t-start 0.5`)

Starts from the input structure with Gaussian noise added at `sigma(t_start)`:

```
coords = clean_structure + randn * sigma_at_t_start
```

Then denoises from step `t_start * num_steps` to `num_steps`. This is how ppd_helper.py is typically used for structure refinement.

### Full Diffusion (`--t-start 0.0`)

Starts from pure noise scaled by `sigma_max`. Protpardelle is unconditional, so this generates a random protein fold (not conditioned on the input structure). High RMSD to any specific target is expected.

### With Guidance (`--guidance-start N`)

After step N, the density reward function is activated. At each guided step, DPS computes the gradient of the density fit loss with respect to the model's denoised prediction (Tweedie) or the noisy state, and adds this gradient to the denoising direction. The density map (`.ccp4`) provides the experimental data.

### Without Guidance (`--no-guidance`)

The density reward is never activated. The model acts purely as a generative prior. With partial diffusion, this produces a structure refined by the model's learned distribution.

## Langevin SDE Noise

The `--langevin` flag enables Song VE-SDE noise injection after each ODE step, matching `ppd_helper_dru_langevin.py` (Song et al., "Score-Based Generative Modeling through SDEs", https://arxiv.org/abs/2011.13456).

After the deterministic Euler step, stochastic noise is added:

```python
# Diffusion coefficient from Song VE-SDE
g = sqrt(sigma_curr^2 * 2 * log(sigma_max / sigma_min))

# Noise injection
xt += Z * g * (1 + langevin_factor)^0.5
```

When `--langevin-factor` > 0, a drift correction is also applied to the score: `score *= (1 + langevin_factor / 2)`. This is implemented by scaling `step_scale` in the sampler config.

Langevin noise improves sample diversity and quality for unconditional sampling. Whether it helps or hurts guidance is an open experimental question.

## EDMSamplerConfig for Protpardelle

The sampler is configured to match ppd_helper.py:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `sigma_data` | derived from noise schedule | Matches `diffusion.noise_schedule` default (10.0) |
| `s_max` | 80.0 | cc89 training config |
| `s_min` | 0.001 | cc89 training config |
| `gamma_0` | 0.0 | Pure ODE (Langevin noise added separately) |
| `step_scale` | 1.0 (or `1 + langevin_factor/2` with drift correction) | ppd_helper default |
| `noise_scale` | 1.0 | No Karras-style noise amplification |
| `augmentation` | False (default) | ppd_helper does not augment |
| `align_to_input` | True | Aligns denoised prediction to input reference |
| `alignment_reverse_diffusion` | True | ppd_helper aligns noisy state onto denoised prediction (Kabsch) |
| `center_each_step` | False | ppd_helper does not re-center at each step |

## Running on HPC

Protpardelle is not a pixi feature. It is added via `PYTHONPATH` from a local clone. Its dependencies (`dm-tree`, `einops`) must be pip-installed or declared in `pyproject.toml`.

```bash
# On Anvil
export PYTHONPATH="/path/to/protpardelle-1c/src:${PYTHONPATH:-}"
sbatch scripts/baseline_protpardelle_guidance_anvil.slurm
```

The SLURM script runs 9 jobs: for each of 6B8X and 1VME, it runs no-guidance ODE, no-guidance Langevin, guidance ODE, and guidance Langevin (all with partial diffusion `t=0.5`). For 6B8X, it also runs a ppd_helper.py baseline as a sanity check.

## Script Arguments

```
python scripts/protpardelle_pure_guidance.py \
    --config-path <yaml>          # Protpardelle YAML config
    --checkpoint-path <pth>       # Protpardelle checkpoint
    --structure <cif/pdb>         # Input structure
    --density <ccp4/mrc>          # Density map (required even for --no-guidance)
    --resolution <float>          # Map resolution in Angstroms
    --output-dir <dir>            # Output directory
    --ensemble-size <int>         # Number of parallel samples (default: 4)
    --t-start <float>             # Partial diffusion start [0,1] (default: 0.0)
    --guidance-start <int>        # Step to begin guidance (default: -1, immediate)
    --align-to-input              # Kabsch align to input each step
    --augmentation                # Random SO(3) augmentation (off for protpardelle)
    --langevin                    # Enable Song VE-SDE noise
    --langevin-factor <float>     # Langevin drift/noise factor (default: 0.0)
    --step-size <float>           # Guidance gradient step size (default: 0.1)
    --use-tweedie                 # Use DataSpaceDPS instead of NoiseSpaceDPS
    --gradient-normalization      # Normalize guidance gradient
    --no-guidance                 # Disable density guidance
```

## References

- Protpardelle: https://github.com/ProteinDesignLab/protpardelle
- Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM): https://arxiv.org/abs/2206.00364
- Song et al. 2021, "Score-Based Generative Modeling through SDEs" (VE-SDE, Langevin): https://arxiv.org/abs/2011.13456
- AlphaFold 3 sampling algorithm: https://www.nature.com/articles/s41586-024-07487-w
