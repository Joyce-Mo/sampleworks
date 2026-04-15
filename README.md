# Sampleworks

> This repository is under active development. Please always use the latest version. If you encounter any problems, please [create an issue on GitHub](https://github.com/diff-use/sampleworks/issues) and include: the PDB ID, the CIF file you used, your density map(s), and log information.

> We would welcome contributions from the community. We are most interested in:
 - new ModelWrappers for additional structure prediction models (especially smaller models which may be more steerable)
 - fast, differentiable modules to allow guidance from other experimental data modalities besides X-ray electron density.

**Sampleworks** is a Python framework for integrating generative biomolecular structure models with experimental data. Read our [blog post](https://diffuse.science/posts/sampleworks/) for an introduction.

## Why sampleworks?

Biomolecular structure prediction and design models are currently trained on single state structures and fail to accurately predict the ensemble of conformations each macromolecule occupies. But there is still hope! Current models show promise in capturing the underlying distribution of realistic macromolecular structures. We want to utilize the prior represented in these models and experimental observations to improve the sampling of the underlying ensemble present in the experiment and use this information to both understand biomolecular function and improve ensemble prediction.

Currently, each structure prediction model has a different implementation, requiring bespoke boilerplate code to plug each model into experimental guidance. Our goal is to resolve this and expand the experimental methods we can provide guidance with. This will open new opportunities for model evaluation directly against experimental data, and help unlock new sources of data for training the next generation of biomolecular structure predictors.

## Installation

**Requirements**: Linux x86-64, CUDA 12, Python ≥ 3.11, < 3.14

### 1. Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### 2. Clone and install

```bash
git clone git@github.com:diff-use/sampleworks.git
cd sampleworks
pixi install -a   # install all environments
```

> **Note**: `pixi install -a` resolves all environments. This (currently) requires CUDA 12 and will fail on machines without it.

Each generative model has its own Pixi environment. Install only what you need:

```bash
pixi install -e boltz      # Boltz-1 / Boltz-2
pixi install -e protenix   # Protenix
pixi install -e rf3        # RosettaFold3
```

### 3. Download model checkpoints

**Boltz-1 and Boltz-2** (stored in `~/.boltz/`):

```bash
pixi run -e boltz python -c "
from boltz.main import download_boltz1, download_boltz2
import pathlib
cache = pathlib.Path('~/.boltz/').expanduser()
download_boltz1(cache)
download_boltz2(cache)
"
```

**Protenix**: checkpoint is downloaded automatically on first use.

**RosettaFold3** (RF3): see the [RC-Foundry repository](https://github.com/RosettaCommons/foundry) for instructions. Default path: `~/.foundry/checkpoints/rf3_foundry_01_24_latest.ckpt`


## Quick Start

Run Boltz-2 pure guidance on the included 1VME example:

```bash
pixi run -e boltz python scripts/boltz2_pure_guidance.py \
    --model-checkpoint ~/.boltz/boltz2_conf.ckpt \
    --structure tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif \
    --density tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4 \
    --resolution 1.8 \
    --output-dir output/boltz2_pure_guidance \
    --guidance-start 130 \
    --ensemble-size 4 \
    --augmentation \
    --align-to-input
```

Output files appear in `output/boltz2_pure_guidance/`: `refined.cif` (final ensemble), `losses.txt`, `trajectory/`, `run.log`. See [`scripts/README.md`](scripts/README.md) for all scripts and arguments.


## Grid Search

`run_grid_search.py` sweeps a model across scalers, ensemble sizes, and gradient weights:

```bash
pixi run -e boltz python run_grid_search.py \
    --proteins proteins.csv \
    --models boltz2 \                # options: boltz1, boltz2, protenix, rf3 (make sure env aligns!)
    --methods "X-RAY DIFFRACTION" \  # only useful for Boltz-2, ignored otherwise
    --scalers pure_guidance \        # options: pure_guidance, fk_steering, or both as space-separated list
    --ensemble-sizes "1 4" \
    --gradient-weights "0.1 0.2" \
    --output-dir grid_search_results \
    --gradient-normalization \       # normalize guidance update magnitude to diffusion update magnitude
    --augmentation \                 # apply random rotations and translations at each step (defaults for inference with AF3-like models)
    --align-to-input                 # align to input structure at each step (required for density guidance to work since it is not rotation/translation invariant)
```

**`proteins.csv` format**

Required columns and format. Supported density map formats: `.ccp4`, `.mrc`, `.map` (not MTZ or SF-CIF yet).
```csv
name,structure,density,resolution
1abc,/data/structures/1abc.cif,/data/maps/1abc.ccp4,2.0
2xyz,/data/structures/2xyz.cif,/data/maps/2xyz.mrc,1.8
```

**Key arguments:**

| Argument | Description | Default |
|---|---|---|
| `--proteins` | CSV with structure/density/resolution columns | required |
| `--models` | Model to run. One of `boltz1`, `boltz2`, `protenix`, `rf3` | required |
| `--scalers` | Guidance method(s) to sweep | `pure_guidance fk_steering` |
| `--ensemble-sizes` | Space-separated values, e.g. `"1 4"` | `"1 2 4 8"` |
| `--gradient-weights` | Space-separated values, e.g. `"0.1 0.2"` | `"0.01 0.1 0.2"` |
| `--methods` | Boltz-2 sampling method (required for boltz2) | `X-RAY DIFFRACTION` |
| `--max-parallel` | Parallel workers (default: number of GPUs) | `auto` |
| `--dry-run` | Print jobs without running them | off |
| `--force-all` | Re-run including already-successful jobs | off |
| `--only-failed` | Re-run only failed jobs | off |
| `--only-missing` | Run only jobs not yet started | off |

Output layout: `grid_search_results/<protein>/<model>[_<method>]/<scaler>/ens<N>_gw<W>/`

> **Note**: Jobs are skipped if a `refined.cif` file already exists in the output directory. Some flags (e.g., `--use-tweedie`, `--gradient-normalization`) are not reflected in the directory structure, so changing them alone won't trigger a re-run. Use `--force-all` to re-run all jobs regardless. This is under active development and will likely change soon.

Instructions for running evaluation and metrics scripts are coming soon.


## Docker

TODO: Docker container documentation


## Development

We use [Pixi](https://pixi.sh/) to manage development environments and dependencies. Each model has its own environment, e.g. `boltz-dev`, `protenix-dev`, `rf3-dev`. To install dev dependencies and run tests:

```bash
pixi install -e [model]-dev    # add pytest, ruff, ty
pixi run -e [model]-dev all-tests  # run tests
pixi run test-all            # run all tests across all environments
```

**Prek hooks** (various formatting, ruff + ty type checking):

```bash
pixi run -e [model]-dev prek install
pixi run -e [model]-dev prek install --hook-type commit-msg
pixi run -e [model]-dev prek run --all-files
```

See [`tests/README.md`](tests/README.md) for full testing instructions.


## macOS (experimental)

To develop on OS X, ensure you have [homebrew](https://brew.sh/) installed and run the following commands to install dependencies:

1. Install hatch and uv
    ```bash
    brew install hatch uv
    ```
2. Move/copy `pyproject-hatch.toml` to `pyproject.toml`
3. Use `uvx hatch run <command>` to run commands. Note the use of `uvx` instead of `uv`
4. Use `uvx hatch run <env>:<command>` to run commands in a specific environment `<env>`.

There are different (and as yet untested) environments for `boltz`. `protenix` won't currently work on a Mac due to
the strict requirement of `triton` which requires an NVIDIA GPU. You may find similar issues with other environments.
Debug as needed.


## Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/) to automate versioning and changelog generation. Format:

```
<type>(<scope>): <summary>
```

Common types: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`, `perf`. A commitizen pre-commit hook validates messages at commit time. See [AGENTS.md](AGENTS.md#release-process) for full details.

# Addition for this fork: Protpardelle wrapper

Architecture Overview

  The pipeline has 5 layers, each with a clear role:

  SLURM script  →  protpardelle_pure_guidance.py  →  PureGuidance
  (orchestrator)
                                                         ↓
                                                AF3EDMSampler (noise schedule +
   Euler steps)
                                                         ↓
                                                ProtpardelleWrapper (model
  interface)

  ---
  1. ProtpardelleWrapper (wrapper.py)

  What it does: Translates between sampleworks' flat atom representation [B,
  n_atoms, 3] and protpardelle's internal [B, L, 37, 3] (residue x atom37)
  representation.

  Libraries: torch (tensors/scatter/gather), biotite (structural biology
  AtomArrays), numpy, jaxtyping (shape annotations)

  Key methods:

  __init__(config_path, checkpoint_path, device)
  - Loads the protpardelle model via protpardelle.core.models.load_model
  - Stores sigma_data (the model's assumed data standard deviation)
  - Initializes mutable self-conditioning state (_struct_self_cond,
  _seq_self_cond)

  featurize(structure) → GenerativeModelInput[ProtpardelleConditioning]
  - Takes an atomworks structure dict, extracts the asymmetric unit as a
  Biotite AtomArray
  - Calls _atomarray_to_atom37() — groups atoms by (chain_id, res_id) and
  places each into its canonical atom37 slot using residue_constants.atom_order
  - Builds the canonical atom mask: for backbone-only models, only N/CA/C/O
  slots; otherwise uses atom37_mask_from_aatype to get all atom slots per
  residue type
  - Computes real_atom_indices — the flat indices within [L*37] where real
  atoms live. This is the key mapping between the two representations
  - Returns x_init (reference coords in flat space) and
  ProtpardelleConditioning

  step(x_t, t, features) → x̂₀
  - Flat → atom37 (scatter): Uses torch.scatter with real_atom_indices to place
   flat atom coords into [B, L, 37, 3] — this is differentiable so gradients
  flow through for guidance
  - Broadcasts sigma t to per-residue [B, L]
  - Passes self-conditioning from the previous step (detached to avoid
  cross-step gradients)
  - Calls the protpardelle model forward pass → gets denoised coords + updated
  self-conditioning
  - Atom37 → flat (gather): Uses torch.gather with same indices to extract real
   atoms back to [B, n_real, 3]

  initialize_from_prior(batch_size, features) → noise
  - Returns randn(batch_size, n_real_atoms, 3) — Gaussian noise in flat atom
  space
  - Resets self-conditioning state

  ---
  2. AF3EDMSampler (edm.py)

  What it does: Implements the EDM (Karras et al. 2022) noise schedule and
  Euler sampling step, following AlphaFold3's Algorithm 18.

  compute_schedule(num_steps=200) → EDMSchedule

  The noise schedule formula:

  σ(t) = σ_data · (s_max^(1/ρ) + t · (s_min^(1/ρ) − s_max^(1/ρ)))^ρ

  where t goes from 0→1 over num_steps. With defaults (σ_data=16, s_max=160,
  s_min=4e-4, ρ=7):
  - Step 0 (t=0): σ = 16 × 160 = 2560 (pure noise)
  - Step 100 (t=0.5): σ ≈ 1.23 (mid-denoise)
  - Step 199 (t≈1): σ ≈ 0.006 (nearly clean)

  Each step also computes:
  - gamma — stochastic noise inflation (0.8 when σ > γ_min=0.2, else 0)
  - t_hat = σ_{t-1} × (1 + γ) — inflated noise level
  - dt = σ_t − t_hat — step size

  step(state, model_wrapper, context, scaler, features) → SamplerStepOutput

  Each Euler step:
  1. Center state, optionally apply random SO(3) augmentation
  2. Add stochastic noise: x_noisy = x + ε, where ε ~ N(0, eps_scale²)
  3. Call model_wrapper.step(x_noisy, t_hat) → get denoised prediction x̂₀
  4. Optionally align x̂₀ to reference frame (Kabsch alignment via reconciler)
  5. Compute denoising direction: δ = (x_noisy − x̂₀) / t_hat
  6. If scaler provided, compute guidance gradient and add to δ
  7. Euler update: x_{next} = x_noisy + step_scale × dt × δ

  EDMSamplerConfig parameters:

  ┌─────────────────────────────┬─────────┬─────────────────────────────────┐
  │          Parameter          │ Default │             Effect              │
  ├─────────────────────────────┼─────────┼─────────────────────────────────┤
  │ sigma_data                  │ 16.0    │ Data distribution std dev —     │
  │                             │         │ scales entire noise schedule    │
  ├─────────────────────────────┼─────────┼─────────────────────────────────┤
  │ s_max                       │ 160.0   │ Max noise ratio — controls      │
  │                             │         │ starting noise level            │
  ├─────────────────────────────┼─────────┼─────────────────────────────────┤
  │ s_min                       │ 4e-4    │ Min noise ratio — controls      │
  │                             │         │ ending noise level              │
  ├─────────────────────────────┼─────────┼─────────────────────────────────┤
  │ p (ρ)                       │ 7.0     │ Schedule curvature — higher =   │
  │                             │         │ more steps at low noise         │
  ├─────────────────────────────┼─────────┼─────────────────────────────────┤
  │ gamma_min                   │ 0.2     │ Below this σ, no stochastic     │
  │                             │         │ noise inflation                 │
  ├─────────────────────────────┼─────────┼─────────────────────────────────┤
  │ gamma_0                     │ 0.8     │ Noise inflation factor          │
  │                             │         │ (S_churn/N in Karras)           │
  ├─────────────────────────────┼─────────┼─────────────────────────────────┤
  │ noise_scale                 │ 1.003   │ Multiplier on stochastic noise  │
  │                             │         │ (S_noise in Karras)             │
  ├─────────────────────────────┼─────────┼─────────────────────────────────┤
  │ step_scale                  │ 1.5     │ Euler step multiplier (AF3 uses │
  │                             │         │  1.5)                           │
  ├─────────────────────────────┼─────────┼─────────────────────────────────┤
  │ augmentation                │ True    │ Random SO(3) rotation each step │
  ├─────────────────────────────┼─────────┼─────────────────────────────────┤
  │ align_to_input              │ True    │ Kabsch-align x̂₀ back to         │
  │                             │         │ reference                       │
  ├─────────────────────────────┼─────────┼─────────────────────────────────┤
  │                             │         │ Align noisy state to x̂₀         │
  │ alignment_reverse_diffusion │ False   │ (Boltz-1 specific, off for      │
  │                             │         │ protpardelle)                   │
  ├─────────────────────────────┼─────────┼─────────────────────────────────┤
  │                             │         │ Rescale guidance gradient to    │
  │ scale_guidance_to_diffusion │ True    │ match diffusion update          │  
  │                             │         │ magnitude                       │
  └─────────────────────────────┴─────────┴─────────────────────────────────┘  
                  
  ---                                            
  3. PureGuidance (pure_guidance.py)
                                    
  What it does: Orchestrates the full sampling loop — featurize → init from
  prior → loop over steps with optional guidance.                              
  
  Parameters:                                                                  
  - ensemble_size — batch of parallel samples
  - num_steps — total diffusion steps (hardcoded 200 in the script)            
  - t_start — if > 0, starts from partial diffusion (noisy version of input
  rather than pure noise)                                                      
  - guidance_t_start — fraction [0,1] of steps after which guidance kicks in   
                                                                               
  The loop (line 108): for i in range(starting_step, num_steps):               
  - Gets step context from sampler schedule                                    
  - If i >= guidance_start, attaches reward function to context                
  - Calls sampler.step() with or without scaler                                
                                                                               
  ---                                            
  4. Step Scalers (step_scalers.py)                                            
                                                                               
  Two DPS (Diffusion Posterior Sampling) variants:
                                                                               
  DataSpaceDPSScaler (default when --use-tweedie is set):                      
  - Computes gradient of reward w.r.t. the denoised prediction x̂₀ only         
  - Fast — doesn't backprop through the model                                  
  - guidance_strength = step_size (constant)     
                                                                               
  NoiseSpaceDPSScaler (default in the script):                                 
  - Computes gradient of reward w.r.t. the noisy input x_t, backpropping       
  through the full model                                                       
  - More accurate but slower and more memory-intensive                         
  - Requires requires_grad=True on the noisy state                             
                                                  
  Both support --gradient-normalization (divides gradient by its L2 norm).