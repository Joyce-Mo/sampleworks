"""
Wrapper for Protpardelle

Follows the ``FlowModelWrapper`` protocol in ``models/protocol.py``
for use in sampleworks guidance pipelines.

Protpardelle is a diffusion model that operates in residue-atom37
coordinate space [B, L, 37, 3]. This wrapper translates between the flat
atom space [B, n_atoms, 3] expected by sampleworks and protpardelle's
internal representation. The steppers are based on ppd_helper.py from Dru.


"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from biotite.structure import AtomArray, AtomArrayStack, infer_elements
from jaxtyping import Float
from loguru import logger
from torch import Tensor

from protpardelle.common import residue_constants
from protpardelle.core.models import load_model
from protpardelle.data.atom import atom37_mask_from_aatype

from sampleworks.eval.structure_utils import get_asym_unit_from_structure
from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.framework_utils import match_batch

# Number of atom slots in atom37 representation
_N_ATOM37 = 37


@dataclass
class ProtpardelleConfig:
    """Configuration for Protpardelle featurization.

    Attributes
    ----------
    ensemble_size : int
        Number of samples to generate (batch dimension of x_init).
    """

    ensemble_size: int = 1


@dataclass(frozen=True, slots=True)
class ProtpardelleConditioning:
    """Static conditioning features for Protpardelle denoising.

    Attributes
    ----------
    seq_mask : Tensor
        Residue validity mask, shape ``[1, L]``.
    residue_index : Tensor
        Residue numbering, shape ``[1, L]``.
    chain_index : Tensor
        Chain identifiers, shape ``[1, L]``.
    atom_mask : Tensor
        Per-atom validity mask in atom37 layout, shape ``[1, L, 37]``.
    aatype : Tensor
        Amino acid types (0-19, 20=UNK), shape ``[1, L]``.
    n_residues : int
        Number of residues.
    n_real_atoms : int
        Number of real (non-dummy) atoms across all residues.
    real_atom_indices : Tensor
        Flat indices of real atoms within the ``[L*37]`` atom37 layout,
        shape ``[n_real_atoms]``. Used for scatter/gather between flat
        and residue-atom37 coordinate spaces.
    true_atom_array : AtomArray | None
        Original structure atom array for alignment reference.
    model_atom_array : AtomArray | None
        Model-space atom array for ``AtomReconciler``.
    """

    seq_mask: Tensor
    residue_index: Tensor
    chain_index: Tensor
    atom_mask: Tensor
    aatype: Tensor
    n_residues: int
    n_real_atoms: int
    real_atom_indices: Tensor
    true_atom_array: AtomArray | None = None
    model_atom_array: AtomArray | None = None


def _atomarray_to_atom37(atom_array: AtomArray) -> dict[str, Any]:
    """Convert Biotite AtomArray to per-residue atom37 representation.

    Groups atoms by ``(chain_id, res_id)`` and places each atom's
    coordinates into the canonical atom37 slot determined by
    ``residue_constants.atom_order``.

    Parameters
    ----------
    atom_array : AtomArray
        Input structure with per-atom annotations.

    Returns
    -------
    dict
        Keys: ``aatype`` (L,), ``atom_positions`` (L, 37, 3),
        ``atom_mask`` (L, 37), ``residue_index`` (L,),
        ``chain_index`` (L,), ``chain_ids`` (list[str]),
        ``res_names`` (list[str]).
    """
    # Group atoms by residue
    residues: list[dict[str, Any]] = []
    current_key: tuple[str, int] | None = None
    for i in range(len(atom_array)):
        key = (str(atom_array.chain_id[i]), int(atom_array.res_id[i]))
        if key != current_key:
            current_key = key
            residues.append(
                {
                    "chain_id": key[0],
                    "res_id": key[1],
                    "res_name": str(atom_array.res_name[i]),
                    "atoms": [],
                }
            )
        residues[-1]["atoms"].append(
            {
                "atom_name": str(atom_array.atom_name[i]),
                "coord": atom_array.coord[i].copy(),
            }
        )

    n_res = len(residues)
    atom_positions = np.zeros((n_res, _N_ATOM37, 3), dtype=np.float32)
    atom_mask = np.zeros((n_res, _N_ATOM37), dtype=np.float32)
    aatype = np.zeros(n_res, dtype=np.int64)
    residue_index = np.zeros(n_res, dtype=np.int64)
    chain_index = np.zeros(n_res, dtype=np.int64)

    chain_id_to_idx: dict[str, int] = {}
    chain_ids: list[str] = []
    res_names: list[str] = []

    for i, res in enumerate(residues):
        three_letter = res["res_name"]
        res_names.append(three_letter)
        one_letter = residue_constants.restype_3to1.get(three_letter, "X")
        aatype[i] = residue_constants.restype_order.get(one_letter, 20)

        residue_index[i] = res["res_id"]
        cid = res["chain_id"]
        chain_ids.append(cid)
        if cid not in chain_id_to_idx:
            chain_id_to_idx[cid] = len(chain_id_to_idx)
        chain_index[i] = chain_id_to_idx[cid]

        for atom in res["atoms"]:
            atom_name = atom["atom_name"].strip()
            if atom_name in residue_constants.atom_order:
                atom_idx = residue_constants.atom_order[atom_name]
                atom_positions[i, atom_idx] = atom["coord"]
                atom_mask[i, atom_idx] = 1.0

    return {
        "aatype": aatype,
        "atom_positions": atom_positions,
        "atom_mask": atom_mask,
        "residue_index": residue_index,
        "chain_index": chain_index,
        "chain_ids": chain_ids,
        "res_names": res_names,
    }


def _build_model_atom_array(
    chain_ids: list[str],
    res_ids: np.ndarray,
    res_names: list[str],
    atom_mask: np.ndarray,
    atom_positions: np.ndarray,
) -> AtomArray:
    """Build a Biotite AtomArray for real atoms in atom37 representation.

    Creates one entry per real atom (``atom_mask > 0``), ordered by
    residue then atom37 index. Used by ``AtomReconciler`` for alignment.

    Parameters
    ----------
    chain_ids : list[str]
        Chain ID per residue, length ``L``.
    res_ids : np.ndarray
        Residue IDs, shape ``(L,)``.
    res_names : list[str]
        3-letter residue names, length ``L``.
    atom_mask : np.ndarray
        Atom validity mask, shape ``(L, 37)``.
    atom_positions : np.ndarray
        Atom coordinates, shape ``(L, 37, 3)``.

    Returns
    -------
    AtomArray
        One entry per real atom with chain_id, res_id, res_name,
        atom_name, element, occupancy, and b_factor annotations.
    """
    n_real = int(atom_mask.sum())
    arr = AtomArray(n_real)

    idx = 0
    for i in range(len(chain_ids)):
        for j in range(_N_ATOM37):
            if atom_mask[i, j] > 0.5:
                arr.chain_id[idx] = chain_ids[i]
                arr.res_id[idx] = int(res_ids[i])
                arr.res_name[idx] = res_names[i]
                arr.atom_name[idx] = residue_constants.atom_types[j]
                arr.coord[idx] = atom_positions[i, j]
                idx += 1

    arr.element = infer_elements(arr)
    arr.set_annotation("occupancy", np.ones(n_real, dtype=np.float32))
    arr.set_annotation("b_factor", np.full(n_real, 20.0, dtype=np.float32))
    return arr


def process_structure_for_protpardelle(
    structure: dict,
    *,
    ensemble_size: int = 1,
) -> dict:
    """Annotate an Atomworks structure with Protpardelle-specific configuration.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary.
    ensemble_size : int
        Number of samples to generate (batch dimension of x_init).

    Returns
    -------
    dict
        Structure dict with ``"_protpardelle_config"`` key added.
    """
    config = ProtpardelleConfig(ensemble_size=ensemble_size)
    return {**structure, "_protpardelle_config": config}


class ProtpardelleWrapper:
    """Wrapper for Protpardelle protein generation model.

    Implements the ``FlowModelWrapper`` protocol for integration with
    sampleworks guidance pipelines. Protpardelle is an EDM-style diffusion
    model that operates in residue-atom37 coordinate space ``[B, L, 37, 3]``.
    This wrapper translates between the flat atom space ``[B, n_atoms, 3]``
    expected by sampleworks and protpardelle's internal representation.

    The ``step()`` method uses differentiable scatter/gather to convert
    between flat and atom37 layouts, enabling gradient-based guidance.
    Self-conditioning state (structure and sequence) is maintained across
    steps and reset on each ``featurize()`` call.

    Parameters
    ----------
    config_path : str | Path
        Path to model YAML configuration file.
    checkpoint_path : str | Path
        Path to model checkpoint (``.pth`` file).
    device : torch.device
        Device for model inference.
    """

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = torch.device(device)
        self.model = load_model(str(config_path), str(checkpoint_path), device=self.device)
        self.model.eval()
        self.sigma_data: float = self.model.sigma_data

        # Mutable self-conditioning state, updated after each step()
        self._struct_self_cond: Tensor | None = None
        self._seq_self_cond: Tensor | None = None

    def featurize(
        self, structure: dict
    ) -> GenerativeModelInput[ProtpardelleConditioning]:
        """Convert atomworks structure dict to Protpardelle features.

        Extracts sequence, coordinates, and atom masks from the input
        structure and builds the conditioning tensors and index mappings
        needed for denoising. Uses the canonical atom37 mask from
        ``atom37_mask_from_aatype`` to determine which atoms the model
        can predict for each residue type.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary with ``"asym_unit"`` key.
            Optionally annotated with ``"_protpardelle_config"`` via
            ``process_structure_for_protpardelle()``.

        Returns
        -------
        GenerativeModelInput[ProtpardelleConditioning]
            Model input with reference coordinates and conditioning.
        """
        # Reset self-conditioning for new structure
        self._struct_self_cond = None
        self._seq_self_cond = None

        config = structure.get("_protpardelle_config", ProtpardelleConfig())
        if isinstance(config, dict):
            config = ProtpardelleConfig(**config)
        ensemble_size = config.ensemble_size

        atom_array_or_stack = get_asym_unit_from_structure(structure)
        true_atom_array: AtomArray = (
            atom_array_or_stack[0]
            if isinstance(atom_array_or_stack, AtomArrayStack)
            else atom_array_or_stack
        )

        # Convert to atom37 representation
        feats = _atomarray_to_atom37(true_atom_array)
        aatype_np = feats["aatype"]
        atom_positions_np = feats["atom_positions"]  # [L, 37, 3]
        residue_index_np = feats["residue_index"]
        n_res = len(aatype_np)

        # Build tensors on device
        aatype_t = torch.tensor(aatype_np, device=self.device, dtype=torch.long).unsqueeze(0)
        seq_mask = torch.ones(1, n_res, device=self.device, dtype=torch.float32)

        # Use sequential 1-indexed residue indices, matching ppd_helper.py
        # (torch.arange(1, Nres+1)). Protpardelle's positional encoding was
        # trained with sequential indices, not raw PDB res_id values which can
        # start at arbitrary numbers or have gaps.
        residue_index_t = torch.arange(
            1, n_res + 1, device=self.device, dtype=torch.long
        ).unsqueeze(0)  # [1, L]

        # Single chain index (all zeros), matching ppd_helper.py behavior.
        # Protpardelle was trained on single-chain proteins.
        chain_index_t = torch.zeros(
            1, n_res, device=self.device, dtype=torch.long
        )  # [1, L]

        # Canonical atom mask: which atoms the model predicts for each residue type
        # For backbone-only models, restrict to backbone atoms
        if self.model.task == "backbone":
            atom_mask_t = torch.zeros(
                1, n_res, _N_ATOM37, device=self.device, dtype=torch.float32
            )
            bb_idxs = residue_constants.backbone_idxs  # [N, CA, C, O]
            atom_mask_t[:, :, bb_idxs] = seq_mask.unsqueeze(-1)
        else:
            atom_mask_t = atom37_mask_from_aatype(aatype_t, seq_mask)  # [1, L, 37]

        # Compute real atom indices within flat [L*37] layout
        atom_mask_flat = atom_mask_t.squeeze(0).reshape(-1)  # [L*37]
        real_atom_indices = torch.nonzero(atom_mask_flat, as_tuple=False).squeeze(-1)
        n_real_atoms = int(real_atom_indices.shape[0])

        # Build model atom array using canonical mask for reconciler
        atom_mask_canonical = atom_mask_t.squeeze(0).cpu().numpy()  # [L, 37]
        model_atom_array = _build_model_atom_array(
            chain_ids=feats["chain_ids"],
            res_ids=residue_index_np,
            res_names=feats["res_names"],
            atom_mask=atom_mask_canonical,
            atom_positions=atom_positions_np,
        )

        conditioning = ProtpardelleConditioning(
            seq_mask=seq_mask,
            residue_index=residue_index_t,
            chain_index=chain_index_t,
            atom_mask=atom_mask_t,
            aatype=aatype_t,
            n_residues=n_res,
            n_real_atoms=n_real_atoms,
            real_atom_indices=real_atom_indices,
            true_atom_array=true_atom_array,
            model_atom_array=model_atom_array,
        )

        # Build x_init from input structure coordinates (flat, real atoms only)
        if len(true_atom_array) == n_real_atoms:
            # Input structure atom count matches model atom count
            x_init = torch.tensor(
                true_atom_array.coord, device=self.device, dtype=torch.float32
            )
        else:
            # Place input coords into atom37 slots, then extract real atoms
            atom37_coords = torch.tensor(
                atom_positions_np, device=self.device, dtype=torch.float32
            )  # [L, 37, 3]
            flat_coords = atom37_coords.reshape(-1, 3)  # [L*37, 3]
            x_init = flat_coords[real_atom_indices]  # [n_real, 3]

            if len(true_atom_array) != n_real_atoms:
                logger.warning(
                    f"Atom count mismatch: input structure has {len(true_atom_array)} atoms "
                    f"but model expects {n_real_atoms}. x_init built from atom37 slot mapping; "
                    "some coordinates may be zero."
                )

        # Center x_init at the origin. The center of mass is computed from
        # real atoms only (x_init already excludes ghost atoms since it was
        # gathered via real_atom_indices). This prevents the structure from
        # drifting far from origin during sampling, which can cause numerical
        # issues in the diffusion process.
        com = x_init.mean(dim=0, keepdim=True)  # [1, 3]
        x_init = x_init - com
        logger.info(
            f"Centered x_init at origin (shifted by {com.squeeze().tolist()})"
        )

        x_init = match_batch(x_init.unsqueeze(0), target_batch_size=ensemble_size).clone()

        return GenerativeModelInput(x_init=x_init, conditioning=conditioning)

    def step(
        self,
        x_t: Float[Tensor, "batch atoms 3"],
        t: Float[Tensor, "*batch"] | float,
        *,
        features: GenerativeModelInput[ProtpardelleConditioning] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        r"""Denoise x_t at noise level t, returning predicted clean structure.

        Translates flat atom coordinates to protpardelle's ``[B, L, 37, 3]``
        layout via differentiable scatter, runs the model forward pass with
        self-conditioning, and gathers real atoms back to flat representation.

        Matches ppd_helper.py's stepper: ghost atoms zeroed, noise_level
        passed as sigma (shape [B] broadcast to [B, L]), sequential
        1-indexed residue_index, run_mpnn_model=False for cc89.

        Parameters
        ----------
        x_t : Float[Tensor, "batch atoms 3"]
            Noisy coordinates in flat atom space.
        t : Float[Tensor, "*batch"] | float
            Noise level (sigma) for EDM denoising.
        features : GenerativeModelInput[ProtpardelleConditioning] | None
            Features from ``featurize()``.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Predicted clean coordinates :math:`\hat{x}_\theta` in flat atom space.
        """
        if features is None or features.conditioning is None:
            raise ValueError("features with conditioning required for step()")

        cond = features.conditioning
        batch_size = x_t.shape[0]
        L = cond.n_residues

        if not isinstance(x_t, torch.Tensor):
            x_t = torch.tensor(x_t, device=self.device, dtype=torch.float32)
        x_t = x_t.to(self.device)

        # Convert t to tensor and broadcast to batch
        if isinstance(t, (int, float)):
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.float32)
        else:
            t_tensor = t.to(device=self.device, dtype=torch.float32)
            if t_tensor.ndim == 0:
                t_tensor = t_tensor.unsqueeze(0)
        t_tensor = match_batch(t_tensor, target_batch_size=batch_size)

        # --- Flat -> atom37 via differentiable scatter ---
        # Build index for scatter: [batch, n_real, 3] -> [batch, L*37, 3]
        real_idx = cond.real_atom_indices  # [n_real]
        scatter_idx = real_idx.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, 3
        )  # [B, n_real, 3]
        noisy_flat = torch.zeros(
            batch_size, L * _N_ATOM37, 3, device=self.device, dtype=x_t.dtype
        )
        noisy_flat = noisy_flat.scatter(1, scatter_idx, x_t)
        noisy_coords = noisy_flat.reshape(batch_size, L, _N_ATOM37, 3)

        # Ghost atoms (positions where atom_mask == 0) must be explicitly zeroed
        # before the model forward pass, matching ppd_helper.py:
        #   mask37 = atom37_mask_from_aatype(self.seq_final, seq_mask).bool()
        #   xt[~mask37] = 0
        ghost_mask = ~cond.atom_mask.bool()  # [1, L, 37], True where ghost
        ghost_mask = ghost_mask.expand(batch_size, -1, -1)  # [B, L, 37]
        noisy_coords[ghost_mask] = 0.0

        # Broadcast sigma to per-residue [B, L].
        # ppd_helper.py passes noise_level as [B] (sigma_curr.expand(B)),
        # which protpardelle's forward() broadcasts internally to [B, L].
        noise_level = t_tensor.unsqueeze(-1).expand(batch_size, L)

        # Expand conditioning to batch size
        seq_mask = match_batch(cond.seq_mask, target_batch_size=batch_size)
        residue_index = match_batch(cond.residue_index, target_batch_size=batch_size)
        chain_index = match_batch(cond.chain_index, target_batch_size=batch_size)

        # Detach self-conditioning to isolate from prior steps' computation graph.
        # Each ensemble member gets its OWN self-conditioning from its own
        # previous prediction, matching ppd_helper.py behavior.
        struct_sc = self._struct_self_cond.detach() if self._struct_self_cond is not None else None
        seq_sc = self._seq_self_cond.detach() if self._seq_self_cond is not None else None

        # Run model forward. run_mpnn_model=False matches ppd_helper.py
        # behavior for cc89 (predict_seq=False). The MPNN doesn't modify
        # denoised coordinates, but running it produces seq_self_cond that
        # would be fed back into subsequent steps, diverging from the
        # reference pipeline.
        denoised_coords, _seq_logprobs, struct_sc_out, seq_sc_out = self.model(
            noisy_coords=noisy_coords,
            noise_level=noise_level,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            struct_self_cond=struct_sc,
            seq_self_cond=seq_sc,
            run_mpnn_model=False,
        )

        # Update self-conditioning state for ALL ensemble members.
        # Each sample gets its own self-conditioning at the next step,
        # matching ppd_helper.py where prev_pred has full batch dimension.
        self._struct_self_cond = struct_sc_out.detach()
        self._seq_self_cond = seq_sc_out.detach()

        # --- Atom37 -> flat via differentiable gather ---
        denoised_flat = denoised_coords.reshape(batch_size, L * _N_ATOM37, 3)
        gather_idx = real_idx.unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, 3
        )  # [B, n_real, 3]
        x0_flat = torch.gather(denoised_flat, 1, gather_idx)  # [B, n_real, 3]

        return x0_flat

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[ProtpardelleConditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Sample from the prior N(0, sigma_max^2 * I) in flat atom space.

        The noise is scaled by sigma_max from the model's native noise
        schedule, matching ppd_helper.py:
            coords = torch.randn(...)
            coords *= self.noise_schedule(1.0)

        Parameters
        ----------
        batch_size : int
            Number of samples.
        features : GenerativeModelInput[ProtpardelleConditioning] | None
            Features from ``featurize()`` to determine atom count.
        shape : tuple[int, ...] | None
            Explicit ``(n_atoms, 3)`` shape if features unavailable.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Scaled Gaussian noise coordinates.

        Raises
        ------
        ValueError
            If neither features nor shape is provided.
        """
        # Reset self-conditioning when starting fresh from prior
        self._struct_self_cond = None
        self._seq_self_cond = None

        # Scale noise by sigma_max from protpardelle's noise schedule at t=1.0
        # (t=1 = max noise in protpardelle's convention, opposite of Karras).
        # Reference: ppd_helper.py line 94-95:
        #   coords = torch.randn(...)
        #   coords *= self.noise_schedule(1.0)
        sigma_max = self.model.sampling_noise_schedule_default(
            torch.tensor(1.0)
        ).to(self.device)

        if shape is not None:
            if len(shape) != 2 or shape[1] != 3:
                raise ValueError("shape must be of the form (num_atoms, 3)")
            return torch.randn((batch_size, *shape), device=self.device) * sigma_max

        if features is None or features.conditioning is None:
            raise ValueError(
                "Either features or shape must be provided to initialize_from_prior()"
            )

        n_real = features.conditioning.n_real_atoms
        return torch.randn((batch_size, n_real, 3), device=self.device) * sigma_max
