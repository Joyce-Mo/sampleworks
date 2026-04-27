"""
Run protpardelle sampling with optional real-space density guidance.

Protpardelle does not yet have a pixi environment or an entry in
``StructurePredictor`` / ``get_model_and_device()``, so this script
instantiates the wrapper directly and runs the sampling loop manually,
matching ppd_helper_dru_langevin.py's Song discretization with Kabsch
alignment.

Sampling (without guidance):
  The model acts as a generative prior. Partial diffusion (--t-start)
  starts from the input structure with noise added at the appropriate
  sigma level, then denoises back to a clean structure.

Guidance (with density):
  Same sampling, but with a density reward function that steers the
  denoising trajectory to fit an experimental electron density map.
  This is the core sampleworks use case per agents.md and readme.md.

Usage (with guidance):
  python scripts/protpardelle_pure_guidance.py \
      --config-path ~/.protpardelle/config.yaml \
      --checkpoint-path ~/.protpardelle/model.pth \
      --structure tests/resources/6b8x/6B8X_single_001_density_input.cif \
      --density tests/resources/6b8x/6B8X_single_001.ccp4 \
      --resolution 1.8 \
      --output-dir output/protpardelle_with_guidance \
      --guidance-start 250 \
      --ensemble-size 4 \
      --t-start 0.5 \
      --align-to-input

Usage (without guidance):
  python scripts/protpardelle_pure_guidance.py \
      --config-path ~/.protpardelle/config.yaml \
      --checkpoint-path ~/.protpardelle/model.pth \
      --structure tests/resources/6b8x/6B8X_single_001_density_input.cif \
      --density tests/resources/6b8x/6B8X_single_001.ccp4 \
      --resolution 1.8 \
      --output-dir output/protpardelle_no_guidance \
      --ensemble-size 4 \
      --t-start 0.5 \
      --no-guidance
"""

import argparse
from pathlib import Path

import torch
from atomworks import parse
from loguru import logger
from tqdm import tqdm

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.core.rewards.real_space_density import (
    RealSpaceRewardFunction,
    setup_scattering_params,
)
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.core.scalers.protocol import GuidanceOutput
from sampleworks.core.scalers.step_scalers import DataSpaceDPSScaler
from sampleworks.eval.structure_utils import process_structure_to_trajectory_input
from sampleworks.models.protpardelle.wrapper import (
    ProtpardelleWrapper,
    process_structure_for_protpardelle,
)
from sampleworks.utils.guidance_script_utils import save_everything
from sampleworks.utils.torch_utils import try_gpu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Protpardelle sampling with optional real-space density guidance"
    )
    # Protpardelle-specific
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to Protpardelle YAML config"
    )
    parser.add_argument(
        "--checkpoint-path", type=str, required=True, help="Path to Protpardelle .pth checkpoint"
    )
    # Generic guidance args
    parser.add_argument("--structure", type=str, required=True, help="Input structure (.cif/.pdb)")
    parser.add_argument("--density", type=str, required=True, help="Input density map (.ccp4/.mrc)")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--resolution", type=float, required=True, help="Map resolution in Angstroms"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto-detect)")
    parser.add_argument("--loss-order", type=int, default=2, choices=[1, 2], help="L1 or L2 loss")
    parser.add_argument("--em", action="store_true", help="Use EM scattering factors")
    parser.add_argument(
        "--ensemble-size", type=int, default=4, help="Number of samples to generate"
    )
    parser.add_argument(
        "--guidance-start",
        type=int,
        default=-1,
        help="Diffusion step at which to start guidance (default: -1, start immediately)",
    )
    parser.add_argument("--augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--align-to-input", action="store_true", help="Align to input structure")
    parser.add_argument(
        "--gradient-normalization", action="store_true", help="Enable gradient normalization"
    )
    # Pure guidance args
    parser.add_argument("--step-size", type=float, default=0.1, help="Gradient step size")
    parser.add_argument("--use-tweedie", action="store_true", help="Use Tweedie formula")
    # Partial diffusion
    parser.add_argument(
        "--t-start",
        type=float,
        default=0.0,
        help="Starting time fraction [0,1] for partial diffusion. "
        "0.0 = full diffusion from pure noise, 0.5 = start halfway. "
        "Matches ppd_helper.py's initialize_partial_noise: adds noise at "
        "sigma(t_start) to the input structure.",
    )
    # Song VE-SDE Langevin noise (ppd_helper_dru_langevin.py, lines 278-281).
    # After the ODE step, adds stochastic noise:
    #   g = sqrt(sigma^2 * 2 * log(sigma_max / sigma_min))
    #   xt += Z * g * (1 + langevin_factor)^0.5
    # Also applies drift correction to the score: score *= (1 + langevin_factor/2)
    # See Song et al. "Score-Based Generative Modeling through SDEs"
    # https://arxiv.org/abs/2011.13456
    parser.add_argument(
        "--langevin",
        action="store_true",
        help="Enable Song VE-SDE Langevin noise injection after each ODE step. "
        "Matches ppd_helper_dru_langevin.py (add_noise=True).",
    )
    parser.add_argument(
        "--langevin-factor",
        type=float,
        default=0.0,
        help="Langevin drift/noise scaling factor. 0 = standard SDE noise, "
        ">0 = amplified noise and drift correction (ppd_helper default: 0).",
    )
    # No-guidance mode
    parser.add_argument(
        "--no-guidance",
        action="store_true",
        help="Run without density guidance (model prior only)",
    )
    return parser.parse_args()


def main(args):
    device = torch.device(args.device) if args.device else try_gpu()
    logger.info(f"Using device: {device}")

    # 500 steps matching ppd_helper.py default (max_steps=500)
    num_steps = 500

    # Load model
    logger.info(f"Loading Protpardelle from {args.checkpoint_path}")
    model_wrapper = ProtpardelleWrapper(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )

    # Load structure and density
    logger.info(f"Loading structure from {args.structure}")
    structure = parse(
        Path(args.structure),
        hydrogen_policy="remove",
        add_missing_atoms=False,
        ccd_mirror_path=None,
    )
    structure = process_structure_for_protpardelle(structure, ensemble_size=args.ensemble_size)

    logger.info(f"Loading density map from {args.density}")
    xmap = XMap.fromfile(args.density, resolution=args.resolution)

    atom_array = structure["asym_unit"]
    scattering_params = setup_scattering_params(em_mode=args.em, device=device)
    selection_mask = atom_array.occupancy > 0

    reward_function = RealSpaceRewardFunction(
        xmap,
        scattering_params,
        selection_mask,
        em=args.em,
        loss_order=args.loss_order,
        device=device,
    )

    # Derive sigma_data for the schedule from protpardelle's own noise schedule
    # function. The noise schedule partial does NOT include sigma_data, so it
    # uses the function's hardcoded default (10.0), which can differ from the
    # model's internal sigma_data (used for EDM preconditioning, may be
    # auto-calculated). We compute from sigma_max / s_max to guarantee match.
    noise_schedule = model_wrapper.model.sampling_noise_schedule_default
    sigma_max = noise_schedule(torch.tensor(1.0)).item()
    s_max = 80.0
    s_min = 0.001
    sigma_data_for_schedule = sigma_max / s_max
    logger.info(
        f"Noise schedule: sigma_max={sigma_max:.2f}, "
        f"sigma_data_for_schedule={sigma_data_for_schedule:.4f}, "
        f"model.sigma_data={model_wrapper.sigma_data:.4f}"
    )

    # Langevin drift correction: ppd_helper_dru_langevin.py line 261 scales
    # the score by (1 + langevin_factor/2). Since score is proportional to
    # step_scale in the Euler update, this is equivalent to scaling step_scale.
    # When langevin_factor=0, drift_step_scale=1.0 (no correction).
    drift_step_scale = 1.0 * (1.0 + args.langevin_factor / 2.0)

    # Stepper config matching ppd_helper_dru_langevin.py's Song discretization:
    #   - gamma_0=0.0: the ODE step itself is deterministic. Langevin noise
    #     (Song VE-SDE) is added AFTER the Euler step via --langevin flag.
    #   - step_scale includes Langevin drift correction when langevin_factor > 0
    #   - alignment_reverse_diffusion=True: ppd_helper aligns noisy state onto
    #     denoised prediction (Kabsch) before Euler step
    #   - center_each_step=False: ppd_helper does not re-center at each step
    stepper = AF3EDMSampler(
        EDMSamplerConfig(
            sigma_data=sigma_data_for_schedule,
            s_max=s_max,
            s_min=s_min,
            gamma_0=0.0,
            step_scale=drift_step_scale,
            noise_scale=1.0,
            device=str(device),
            augmentation=args.augmentation,
            align_to_input=args.align_to_input,
            alignment_reverse_diffusion=True,
            center_each_step=False,
        )
    )

    # Precompute Langevin diffusion coefficient for Song VE-SDE noise.
    # g = sqrt(sigma^2 * 2 * log(sigma_max / sigma_min))
    # ppd_helper_dru_langevin.py line 280 uses log(800/0.01). The ratio
    # 800/0.01 = sigma_max / sigma_min from the noise schedule.
    import math
    sigma_min_schedule = sigma_data_for_schedule * s_min
    langevin_log_ratio = 2.0 * math.log(sigma_max / sigma_min_schedule)
    if args.langevin:
        logger.info(
            f"Langevin enabled: log_ratio={langevin_log_ratio:.2f}, "
            f"factor={args.langevin_factor}, "
            f"drift_step_scale={drift_step_scale:.4f}"
        )

    # Step scaler (only used when guidance is active)
    if args.use_tweedie:
        step_scaler = DataSpaceDPSScaler(
            step_size=args.step_size,
            gradient_normalization=args.gradient_normalization,
        )
    else:
        from sampleworks.core.scalers.step_scalers import NoiseSpaceDPSScaler

        step_scaler = NoiseSpaceDPSScaler(
            step_size=args.step_size,
            gradient_normalization=args.gradient_normalization,
        )

    # Guidance start step
    if args.no_guidance:
        guidance_start = num_steps  # never start guidance
    else:
        guidance_start = args.guidance_start if args.guidance_start > 0 else 0

    # Compute starting step from t_start fraction
    starting_step = int(args.t_start * num_steps)

    mode = "WITHOUT" if args.no_guidance else "WITH"
    logger.info(
        f"Running Protpardelle {mode} guidance, "
        f"steps {starting_step}-{num_steps}, "
        f"guidance from step {guidance_start}"
    )

    # Featurize and build initial state
    features = model_wrapper.featurize(structure)
    ensemble_size = args.ensemble_size

    # Build schedule
    schedule = stepper.compute_schedule(num_steps=num_steps)

    # Initialize coordinates
    if starting_step > 0:
        # Partial diffusion: start from the input structure with noise added
        # at sigma(t_start), matching ppd_helper.py's initialize_partial_noise
        # which calls noise_coords(struct, sigma_at_t, atom_mask).
        sigma_at_start = noise_schedule(
            torch.tensor(1.0 - starting_step / num_steps)
        ).to(device)
        logger.info(
            f"Partial diffusion from step {starting_step}, "
            f"sigma={sigma_at_start.item():.2f}"
        )
        # x_init is the centered input structure from featurize()
        clean_coords = features.x_init  # [ensemble, n_atoms, 3]
        noise = torch.randn_like(clean_coords) * sigma_at_start
        coords = clean_coords + noise
    else:
        # Full diffusion from pure noise scaled by sigma_max
        coords = torch.as_tensor(
            model_wrapper.initialize_from_prior(
                batch_size=ensemble_size,
                features=features,
            ),
        )

    # Set up alignment and reward infrastructure
    processed_structure = process_structure_to_trajectory_input(
        structure=structure,
        coords_from_prior=coords,
        features=features,
        ensemble_size=ensemble_size,
    )
    reconciler = processed_structure.reconciler.to(coords.device)
    reward_inputs = processed_structure.to_reward_inputs(device=coords.device)

    trajectory_denoised: list[torch.Tensor] = []
    trajectory_next_step: list[torch.Tensor] = []
    losses: list[float | None] = []

    # Sampling loop
    for i in tqdm(range(starting_step, num_steps)):
        context = stepper.get_context_for_step(i, schedule)
        apply_guidance = i >= guidance_start and not args.no_guidance

        if apply_guidance:
            context = context.with_reward(reward_function, reward_inputs)

        context = context.with_reconciler(
            reconciler=reconciler,
            alignment_reference=processed_structure.input_coords,
        )

        step_output = stepper.step(
            state=coords,
            model_wrapper=model_wrapper,
            context=context,
            scaler=step_scaler if apply_guidance else None,
            features=features,
        )

        coords = step_output.state

        # Song VE-SDE Langevin noise injection (ppd_helper_dru_langevin.py
        # lines 278-281). Applied AFTER the ODE Euler step:
        #   g = sqrt(sigma_curr^2 * 2 * log(sigma_max / sigma_min))
        #   xt += Z * g * (1 + langevin_factor)^0.5
        if args.langevin:
            sigma_curr = context.t  # sigma at current step
            g = (sigma_curr**2 * langevin_log_ratio).sqrt()
            Z = torch.randn_like(coords)
            coords = coords + Z * g * (1.0 + args.langevin_factor) ** 0.5

        trajectory_next_step.append(coords.clone().cpu())

        if step_output.denoised is not None:
            trajectory_denoised.append(step_output.denoised.clone().cpu())

        if step_output.loss is not None:
            losses.append(step_output.loss.mean().item())
        else:
            losses.append(None)

    # Save results
    metadata: dict = {"trajectory_denoised": trajectory_denoised}
    if reconciler.has_mismatch and processed_structure.model_atom_array is not None:
        metadata["model_atom_array"] = processed_structure.model_atom_array

    result = GuidanceOutput(
        structure=structure,
        final_state=coords,
        trajectory=trajectory_next_step,
        losses=losses,
        metadata=metadata,
    )

    model_atom_array = result.metadata.get("model_atom_array") if result.metadata else None

    save_everything(
        args.output_dir,
        losses,
        result.structure,
        trajectory_denoised,
        list(result.trajectory) if result.trajectory else [],
        "pure_guidance",
        final_state=torch.as_tensor(result.final_state),
        model_atom_array=model_atom_array,
    )


if __name__ == "__main__":
    main(parse_args())
