"""
Run pure guidance with real-space density reward on the Protpardelle model.

Protpardelle does not yet have a pixi environment or an entry in
``StructurePredictor`` / ``get_model_and_device()``, so this script
instantiates the wrapper directly and wires up the guidance pipeline manually
(mirroring the logic in ``guidance_script_utils._run_guidance``).

Usage (with guidance):
  python scripts/protpardelle_pure_guidance.py \
      --config-path ~/.protpardelle/config.yaml \
      --checkpoint-path ~/.protpardelle/model.pth \
      --structure tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif \
      --density tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4 \
      --resolution 1.8 \
      --output-dir output/protpardelle_guidance \
      --guidance-start 130 \
      --ensemble-size 4 \
      --augmentation --align-to-input

Usage (without guidance — unconditional sampling):
  python scripts/protpardelle_pure_guidance.py \
      --config-path ~/.protpardelle/config.yaml \
      --checkpoint-path ~/.protpardelle/model.pth \
      --structure tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif \
      --density tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4 \
      --resolution 1.8 \
      --output-dir output/protpardelle_no_guidance \
      --ensemble-size 4 \
      --no-guidance
"""

import argparse
from pathlib import Path

import torch
from atomworks import parse
from loguru import logger

from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.core.rewards.real_space_density import (
    RealSpaceRewardFunction,
    setup_scattering_params,
)
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.core.scalers.pure_guidance import PureGuidance
from sampleworks.core.scalers.step_scalers import DataSpaceDPSScaler
from sampleworks.models.protpardelle.wrapper import (
    ProtpardelleWrapper,
    process_structure_for_protpardelle,
)
from sampleworks.utils.guidance_script_utils import save_everything
from sampleworks.utils.torch_utils import try_gpu


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pure guidance refinement with Protpardelle and real-space density"
    )
    # Protpardelle-specific
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to Protpardelle YAML config"
    )
    parser.add_argument(
        "--checkpoint-path", type=str, required=True, help="Path to Protpardelle .pth checkpoint"
    )
    # Generic guidance args (mirrors add_generic_args)
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
    # No-guidance mode
    parser.add_argument(
        "--no-guidance",
        action="store_true",
        help="Run unconditional sampling (no density guidance)",
    )
    return parser.parse_args()


def main(args):
    device = torch.device(args.device) if args.device else try_gpu()
    logger.info(f"Using device: {device}")

    # 500 steps like what dru uses
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
    # function. The model's noise schedule (make_sampling_noise_schedule) does
    # NOT pass sigma_data to diffusion.noise_schedule, so it uses the function's
    # hardcoded default (10.0), which can differ from the model's internal
    # sigma_data (used for EDM preconditioning). Using model_wrapper.sigma_data
    # here would produce a DIFFERENT sigma trajectory than ppd_helper, because
    # auto_calc_sigma_data can change the model's sigma_data at training time.
    # We compute it from sigma_max / s_max to guarantee exact match.
    sigma_max = model_wrapper.model.sampling_noise_schedule_default(
        torch.tensor(1.0)
    ).item()
    s_max = 80.0
    s_min = 0.001
    sigma_data_for_schedule = sigma_max / s_max
    logger.info(
        f"Noise schedule: sigma_max={sigma_max:.2f}, "
        f"sigma_data_for_schedule={sigma_data_for_schedule:.4f}, "
        f"model.sigma_data={model_wrapper.sigma_data:.4f}"
    )

    # Stepper config matching ppd_helper.py's ODE sampling behavior:
    #   - gamma_0=0.0: pure ODE (no stochastic noise injection)
    #   - step_scale=1.0: ppd_helper default (AF3 uses 1.5)
    #   - alignment_reverse_diffusion=True: ppd_helper aligns noisy state onto
    #     denoised prediction before Euler step
    stepper = AF3EDMSampler(
        EDMSamplerConfig(
            sigma_data=sigma_data_for_schedule,
            s_max=s_max,
            s_min=s_min,
            gamma_0=0.0,
            step_scale=1.0,
            noise_scale=1.0,
            device=str(device),
            augmentation=args.augmentation,
            align_to_input=args.align_to_input,
            alignment_reverse_diffusion=True,
        )
    )

    # Step scaler
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

    # Guidance start
    if args.no_guidance:
        guidance_t_start = 1.0  # never start guidance
    else:
        guidance_t_start = args.guidance_start / num_steps if args.guidance_start > 0 else 0.0

    guidance = PureGuidance(
        ensemble_size=args.ensemble_size,
        num_steps=num_steps,
        guidance_t_start=guidance_t_start,
    )

    mode = "WITHOUT" if args.no_guidance else "WITH"
    logger.info(f"Running Protpardelle {mode} guidance")

    result = guidance.sample(
        structure=structure,
        model=model_wrapper,
        sampler=stepper,
        step_scaler=step_scaler,
        reward=reward_function,
    )

    # Save
    model_atom_array = result.metadata.get("model_atom_array") if result.metadata else None
    traj_denoised = result.metadata.get("trajectory_denoised", []) if result.metadata else []
    traj_next_step = list(result.trajectory) if result.trajectory else []
    losses = result.losses if result.losses else []

    save_everything(
        args.output_dir,
        losses,
        result.structure,
        traj_denoised,
        traj_next_step,
        "pure_guidance",
        final_state=torch.as_tensor(result.final_state),
        model_atom_array=model_atom_array,
    )


if __name__ == "__main__":
    main(parse_args())
