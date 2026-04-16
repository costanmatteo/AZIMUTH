"""
Mathematical computation of process chain reliability F.

This module computes reliability F from process chain trajectories using
adaptive targets that account for process dependencies.

Formula:
    For each process i:
        Q_i = exp(-(output_i - τ_i)² / s_i)

    Where τ_i is an adaptive target that depends on upstream process outputs.

    Overall reliability:
        F = Σ(w_i × Q_i) / Σ(w_i)
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union

from .process_targets import PROCESS_CONFIGS, PROCESS_ORDER


def _apply_adaptive_mode(delta, coeff, mode, mode_params):
    """
    Compute the adaptive target adjustment for a single upstream variable.

    Args:
        delta: torch.Tensor — (upstream_out - baseline)
        coeff: float — linear coefficient
        mode: str — one of 'linear', 'polynomial', 'power', 'softplus', 'deadband', 'tanh'
        mode_params: dict — mode-specific params for this upstream variable

    Returns:
        torch.Tensor — adjustment to add to target
    """
    if mode == 'linear':
        return coeff * delta

    if mode == 'polynomial':
        coeff2 = mode_params.get('coeff2', 0.0)
        return coeff * delta + coeff2 * delta ** 2

    if mode == 'power':
        alpha = mode_params.get('alpha', 0.5)
        return coeff * torch.sign(delta) * (torch.abs(delta) + 1e-8) ** alpha

    if mode == 'softplus':
        k = mode_params.get('k', 2.0)
        return (1.0 / k) * torch.log(1.0 + torch.exp(k * coeff * delta))

    if mode == 'deadband':
        band = mode_params.get('band', 0.0)
        abs_delta = torch.abs(delta)
        return coeff * torch.clamp(abs_delta - band, min=0.0) * torch.sign(delta)

    if mode == 'tanh':
        max_shift = mode_params.get('max_shift', 1.0)
        return max_shift * torch.tanh(coeff * delta / max_shift)

    raise ValueError(f"Unknown adaptive_mode: '{mode}'. "
                     f"Expected one of: linear, polynomial, power, softplus, deadband, tanh")


class ReliabilityFunction:
    """
    Computes reliability F from process chain trajectories.

    Supports both torch tensors (for differentiable training) and numpy arrays.
    """

    def __init__(self,
                 process_configs: Dict = None,
                 process_order: list = None,
                 device: str = 'cpu'):
        """
        Args:
            process_configs: Process-specific targets and weights.
                            If None, uses default PROCESS_CONFIGS.
            process_order: Order of processes for dependency resolution.
                          If None, uses default PROCESS_ORDER.
            device: Torch device for computations.
        """
        self.process_configs = process_configs or PROCESS_CONFIGS
        self.process_order = process_order or PROCESS_ORDER
        self.device = device

    def compute_reliability(self,
                           trajectory: Dict,
                           return_quality_scores: bool = False,
                           use_sampled_outputs: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute reliability F for a trajectory.

        Args:
            trajectory: Dict mapping process_name to:
                {
                    'inputs': tensor (batch, input_dim),
                    'outputs_mean': tensor (batch, output_dim),
                    'outputs_var': tensor (batch, output_dim),  # optional
                    'outputs_sampled': tensor (batch, output_dim)  # optional
                }
            return_quality_scores: If True, also return per-process Q_i scores.
            use_sampled_outputs: If True, use 'outputs_sampled' if available,
                                otherwise use 'outputs_mean'.

        Returns:
            F: Reliability score (batch,) or scalar
            quality_scores: Dict of per-process Q_i (if return_quality_scores=True)
        """
        # Extract outputs from trajectory
        outputs = {}
        for process_name, data in trajectory.items():
            if use_sampled_outputs and 'outputs_sampled' in data:
                output = data['outputs_sampled']
            else:
                output = data['outputs_mean']

            # Ensure tensor
            if isinstance(output, np.ndarray):
                output = torch.tensor(output, dtype=torch.float32, device=self.device)

            # Keep multi-dimensional output as-is; ensure at least 2D for uniform handling
            if output.dim() == 1:
                output = output.unsqueeze(-1)  # (batch,) → (batch, 1)
            outputs[process_name] = output  # (batch, output_dim)

        # Compute adaptive targets and quality scores
        adaptive_targets = {}
        quality_scores = {}

        for process_name in self.process_order:
            if process_name not in outputs:
                continue

            config = self.process_configs.get(process_name, {})
            output = outputs[process_name]

            # Compute adaptive target (broadcastable to (batch, output_dim))
            target = self._compute_adaptive_target(process_name, outputs, config)
            adaptive_targets[process_name] = target

            # Get per-dimension scale; wrap scalar in list for uniform handling
            scales = config.get('scale', 1.0)
            if not isinstance(scales, list):
                scales = [scales]
            scale_t = torch.tensor(scales, dtype=torch.float32, device=self.device)  # (output_dim,)

            # Compute per-dimension quality and average across output dims
            per_dim_q = torch.exp(-((output - target) ** 2) / scale_t)  # (batch, output_dim)
            quality = per_dim_q.mean(dim=-1)  # (batch,)

            quality_scores[process_name] = quality

        # Compute weighted average reliability
        F = self._compute_weighted_average(quality_scores)

        if return_quality_scores:
            return F, quality_scores
        return F

    def _compute_adaptive_target(self,
                                 process_name: str,
                                 outputs: Dict[str, torch.Tensor],
                                 config: Dict) -> Union[torch.Tensor, float]:
        """
        Compute adaptive target for a process based on upstream outputs.

        Returns a value broadcastable to (batch, output_dim):
        - No adaptive coefficients: tensor of shape (output_dim,) from base_target list
        - With adaptive coefficients: scalar or (batch, 1) tensor that broadcasts
          uniformly across all output dimensions.
        """
        base_target = config.get('base_target', 0.0)

        # Normalize base_target to a tensor (output_dim,)
        if isinstance(base_target, list):
            base_target_t = torch.tensor(base_target, dtype=torch.float32, device=self.device)
        else:
            base_target_t = torch.tensor([base_target], dtype=torch.float32, device=self.device)

        adaptive_coeffs = config.get('adaptive_coefficients', {})
        adaptive_baselines = config.get('adaptive_baselines', {})

        if not adaptive_coeffs:
            return base_target_t  # (output_dim,) — broadcasts to (batch, output_dim)

        # Read adaptive mode and per-upstream mode params
        mode = config.get('adaptive_mode', 'linear')
        adaptive_coefficients2 = config.get('adaptive_coefficients2', {})
        adaptive_power = config.get('adaptive_power', {})
        adaptive_sharpness = config.get('adaptive_sharpness', {})
        adaptive_band = config.get('adaptive_band', {})
        adaptive_max_shift = config.get('adaptive_max_shift', {})

        # Adaptive case: compute a single scalar target for all output dimensions.
        # Average base_target across output dims to get a scalar starting point.
        target = base_target_t.mean()

        for upstream_name, coeff in adaptive_coeffs.items():
            if upstream_name in outputs:
                upstream_out = outputs[upstream_name]
                # Average across output dims if multi-dimensional → scalar per sample
                if upstream_out.dim() > 1:
                    upstream_out = upstream_out.mean(dim=-1)  # (batch,)

                baseline = adaptive_baselines.get(upstream_name, 0.0)
                delta = upstream_out - baseline

                # Build mode_params for this upstream variable
                mode_params = {}
                if mode == 'polynomial':
                    mode_params['coeff2'] = adaptive_coefficients2.get(upstream_name, 0.0)
                elif mode == 'power':
                    mode_params['alpha'] = adaptive_power.get(upstream_name, 0.5)
                elif mode == 'softplus':
                    mode_params['k'] = adaptive_sharpness.get(upstream_name, 2.0)
                elif mode == 'deadband':
                    mode_params['band'] = adaptive_band.get(upstream_name, 0.0)
                elif mode == 'tanh':
                    mode_params['max_shift'] = adaptive_max_shift.get(upstream_name, 1.0)

                adjustment = _apply_adaptive_mode(delta, coeff, mode, mode_params)
                target = target + adjustment  # scalar + (batch,) → (batch,)

        # Ensure result broadcasts to (batch, output_dim): (batch,) → (batch, 1)
        if isinstance(target, torch.Tensor) and target.dim() >= 1:
            target = target.unsqueeze(-1)

        return target

    def _compute_weighted_average(self, quality_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted average of quality scores.
        """
        total_weighted_quality = 0.0
        total_weight = 0.0

        for process_name, quality in quality_scores.items():
            weight = self.process_configs.get(process_name, {}).get('weight', 1.0)
            # Handle list weights (multi-output): average to get per-process scalar weight
            if isinstance(weight, list):
                weight = sum(weight) / len(weight)
            total_weighted_quality = total_weighted_quality + quality * weight
            total_weight += weight

        if total_weight > 0:
            F = total_weighted_quality / total_weight
        else:
            F = torch.tensor(0.0, device=self.device)

        return F

    def compute_target_reliability(self, target_trajectory: Dict) -> torch.Tensor:
        """
        Compute F* (target reliability) for a target trajectory.
        """
        trajectory = {}
        for process_name, data in target_trajectory.items():
            outputs = data.get('outputs', data.get('outputs_mean'))
            if isinstance(outputs, np.ndarray):
                outputs = torch.tensor(outputs, dtype=torch.float32, device=self.device)

            trajectory[process_name] = {
                'outputs_mean': outputs,
                'outputs_sampled': outputs,
            }

        return self.compute_reliability(trajectory)


def compute_reliability(trajectory: Dict,
                       return_quality_scores: bool = False,
                       device: str = 'cpu') -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
    """
    Convenience function to compute reliability F.
    """
    rf = ReliabilityFunction(device=device)
    return rf.compute_reliability(trajectory, return_quality_scores=return_quality_scores)


class ShekelReliabilityFunction:
    """
    Alternative reliability function using a single-peak adaptive Shekel bump.

    F(o) = 1 / (1 + sum_{t,k} c_{t,k} * (o_{t,k} - a_{t,k}_eff)^2)

    where a_{t,k}_eff uses the same adaptive target mechanism as ReliabilityFunction,
    and c_{t,k} = s / sigma_{t,k}^2 is calibrated from the output distribution.

    F ∈ (0, 1], F = 1 only when all outputs hit their adaptive peak centers exactly.
    """

    def __init__(self,
                 process_configs: Dict = None,
                 process_order: list = None,
                 device: str = 'cpu',
                 s: float = 1.0):
        """
        Args:
            process_configs: Process-specific shekel configs (with 'shekel_center',
                             'shekel_sigma', and the usual adaptive keys).
                             If None, uses default PROCESS_CONFIGS.
            process_order: Order of processes for dependency resolution.
                           If None, uses default PROCESS_ORDER.
            device: Torch device for computations.
            s: Global width hyperparameter applied to all processes
               (c_{t,k} = s / sigma_{t,k}^2).
        """
        self.process_configs = process_configs or PROCESS_CONFIGS
        self.process_order = process_order or PROCESS_ORDER
        self.device = device
        self.s = float(s)

    def compute_reliability(self,
                            trajectory: Dict,
                            return_per_process_contributions: bool = False,
                            use_sampled_outputs: bool = True
                            ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute the Shekel-bump reliability F for a trajectory.

        Args:
            trajectory: Dict mapping process_name to:
                {
                    'inputs': tensor (batch, input_dim),
                    'outputs_mean': tensor (batch, output_dim),
                    'outputs_var': tensor (batch, output_dim),   # optional
                    'outputs_sampled': tensor (batch, output_dim)  # optional
                }
            return_per_process_contributions: If True, also return a dict mapping
                process_name to tensor(batch,) of its summed weighted
                squared-distance contribution to the denominator.
            use_sampled_outputs: If True, use 'outputs_sampled' if available,
                                 otherwise use 'outputs_mean'.

        Returns:
            F: tensor (batch,), values in (0, 1]
            contributions (optional): dict {process_name: tensor(batch,)}
        """
        # Extract outputs from trajectory
        outputs = {}
        for process_name, data in trajectory.items():
            if use_sampled_outputs and 'outputs_sampled' in data:
                output = data['outputs_sampled']
            else:
                output = data['outputs_mean']

            if isinstance(output, np.ndarray):
                output = torch.tensor(output, dtype=torch.float32, device=self.device)

            if output.dim() == 1:
                output = output.unsqueeze(-1)  # (batch,) → (batch, 1)
            outputs[process_name] = output  # (batch, output_dim)

        contributions: Dict[str, torch.Tensor] = {}
        total_sq_sum = None  # running (batch,) tensor

        for process_name in self.process_order:
            if process_name not in outputs:
                continue

            config = self.process_configs.get(process_name, {})
            output = outputs[process_name]  # (batch, output_dim)
            output_dim = output.shape[-1]

            # Adaptive center, broadcastable to (batch, output_dim)
            center = self._compute_adaptive_center(process_name, outputs, config)

            # Width coefficients c_{t,k} = s / sigma_{t,k}^2
            sigmas = config.get('shekel_sigma', None)
            if sigmas is None:
                sigmas = [1.0] * output_dim
            if not isinstance(sigmas, (list, tuple)):
                sigmas = [sigmas]
            sigma_t = torch.tensor(sigmas, dtype=torch.float32, device=self.device)
            c_t = self.s / (sigma_t ** 2 + 1e-8)  # (output_dim,)

            # Per-dimension weighted squared distance
            sq = c_t * (output - center) ** 2  # (batch, output_dim)
            process_contrib = sq.sum(dim=-1)  # (batch,)

            contributions[process_name] = process_contrib

            if total_sq_sum is None:
                total_sq_sum = process_contrib
            else:
                total_sq_sum = total_sq_sum + process_contrib

        if total_sq_sum is None:
            total_sq_sum = torch.tensor(0.0, device=self.device)

        F = 1.0 / (1.0 + total_sq_sum)

        if return_per_process_contributions:
            return F, contributions
        return F

    @staticmethod
    def _coeff_to_matrix(coeff_spec, d_t, d_j, device='cpu'):
        """
        Normalize an adaptive coefficient specification to a (d_t, d_j) matrix.

        Accepted formats:
            - scalar float:              broadcast to all (k_t, k_j) pairs
            - list[float] of length d_j: per-upstream-dim, same for all target dims
            - list[list[float]] (d_t×d_j): full coupling matrix
        """
        if isinstance(coeff_spec, (int, float)):
            return torch.full((d_t, d_j), float(coeff_spec),
                              dtype=torch.float32, device=device)
        if isinstance(coeff_spec, list):
            if len(coeff_spec) > 0 and isinstance(coeff_spec[0], list):
                return torch.tensor(coeff_spec, dtype=torch.float32, device=device)
            # flat list → (d_j,), broadcast to every target dim
            v = torch.tensor(coeff_spec, dtype=torch.float32, device=device)
            return v.unsqueeze(0).expand(d_t, -1)
        return torch.full((d_t, d_j), float(coeff_spec),
                          dtype=torch.float32, device=device)

    def _compute_adaptive_center(self,
                                 process_name: str,
                                 outputs: Dict[str, torch.Tensor],
                                 config: Dict) -> torch.Tensor:
        """
        Compute the effective adaptive peak center for a process.

        Each upstream output dimension independently influences each target
        dimension through a coefficient matrix (d_t × d_j).

        adaptive_coefficients[upstream] can be:
            - scalar:              same coeff for every (k_t, k_j) pair
            - list of length d_j:  per-upstream-dim, broadcast to all target dims
            - list-of-lists d_t×d_j: full coupling matrix

        adaptive_baselines[upstream] can be:
            - scalar:              same baseline for every upstream dim
            - list of length d_j:  per-upstream-dim baseline

        Returns:
            (d_t,) when no adaptive coefficients are present, or
            (batch, d_t) when upstream-dependent shifts are added.
        """
        base_center = config.get('shekel_center', 0.0)

        if isinstance(base_center, list):
            base_center_t = torch.tensor(base_center, dtype=torch.float32, device=self.device)
        else:
            base_center_t = torch.tensor([base_center], dtype=torch.float32, device=self.device)

        d_t = base_center_t.shape[0]

        adaptive_coeffs = config.get('adaptive_coefficients', {})
        adaptive_baselines = config.get('adaptive_baselines', {})

        if not adaptive_coeffs:
            return base_center_t  # (d_t,) — broadcasts to (batch, d_t)

        mode = config.get('adaptive_mode', 'linear')
        adaptive_coefficients2 = config.get('adaptive_coefficients2', {})
        adaptive_power = config.get('adaptive_power', {})
        adaptive_sharpness = config.get('adaptive_sharpness', {})
        adaptive_band = config.get('adaptive_band', {})
        adaptive_max_shift = config.get('adaptive_max_shift', {})

        # Preserve per-dim base centers — shifts will be added per-dim too.
        center = base_center_t  # (d_t,)

        for upstream_name, coeff_spec in adaptive_coeffs.items():
            if upstream_name not in outputs:
                continue

            upstream_out = outputs[upstream_name]  # (batch, d_j)
            if upstream_out.dim() == 1:
                upstream_out = upstream_out.unsqueeze(-1)
            d_j = upstream_out.shape[-1]
            batch_size = upstream_out.shape[0]

            # Per-upstream-dim baseline → (d_j,)
            baseline = adaptive_baselines.get(upstream_name, 0.0)
            if isinstance(baseline, list):
                baseline_t = torch.tensor(baseline, dtype=torch.float32, device=self.device)
            else:
                baseline_t = torch.full((d_j,), float(baseline),
                                        dtype=torch.float32, device=self.device)

            delta = upstream_out - baseline_t  # (batch, d_j)

            # Mode params (per-upstream, shared across dims)
            mode_params = {}
            if mode == 'polynomial':
                mode_params['coeff2'] = adaptive_coefficients2.get(upstream_name, 0.0)
            elif mode == 'power':
                mode_params['alpha'] = adaptive_power.get(upstream_name, 0.5)
            elif mode == 'softplus':
                mode_params['k'] = adaptive_sharpness.get(upstream_name, 2.0)
            elif mode == 'deadband':
                mode_params['band'] = adaptive_band.get(upstream_name, 0.0)
            elif mode == 'tanh':
                mode_params['max_shift'] = adaptive_max_shift.get(upstream_name, 1.0)

            # Coefficient matrix (d_t, d_j)
            coeff_matrix = self._coeff_to_matrix(coeff_spec, d_t, d_j, self.device)

            # Accumulate shift (batch, d_t):
            # shift[:, k_t] = Σ_{k_j} r(delta[:, k_j], coeff_matrix[k_t, k_j], ...)
            shift = torch.zeros(batch_size, d_t, dtype=torch.float32, device=self.device)
            for k_j in range(d_j):
                delta_kj = delta[:, k_j]  # (batch,)
                for k_t in range(d_t):
                    c = float(coeff_matrix[k_t, k_j])
                    if abs(c) < 1e-12:
                        continue
                    adj = _apply_adaptive_mode(delta_kj, c, mode, mode_params)
                    shift[:, k_t] = shift[:, k_t] + adj

            center = center + shift  # (d_t,) + (batch, d_t) → (batch, d_t)

        return center

    def compute_target_reliability(self, target_trajectory: Dict) -> torch.Tensor:
        """
        Compute F* (target reliability) for a target trajectory.
        """
        trajectory = {}
        for process_name, data in target_trajectory.items():
            outputs = data.get('outputs', data.get('outputs_mean'))
            if isinstance(outputs, np.ndarray):
                outputs = torch.tensor(outputs, dtype=torch.float32, device=self.device)

            trajectory[process_name] = {
                'outputs_mean': outputs,
                'outputs_sampled': outputs,
            }

        return self.compute_reliability(trajectory)


def calibrate_shekel_configs(process_configs: Dict,
                             process_order: list,
                             calibration_trajectories: list,
                             s: float = 1.0) -> Dict:
    """
    Build a Shekel-compatible process_configs dict from calibration trajectories.

    For each process, computes the per-dimension mean and standard deviation of
    its outputs across the aggregated calibration dataset, and stores them as
    'shekel_center' and 'shekel_sigma'. All existing adaptive-related keys are
    copied over unchanged.

    Args:
        process_configs: Original process_configs (as used by ReliabilityFunction).
        process_order: List of process names.
        calibration_trajectories: List of trajectory dicts, each of the form
            {process_name: {'outputs_mean': tensor(batch, d_t), ...}}.
        s: Global width hyperparameter (stored for reference; consumed by
           ShekelReliabilityFunction.__init__).

    Returns:
        A new process_configs dict where each process has 'shekel_center' and
        'shekel_sigma' populated from the empirical output distribution.
    """
    # Keys to carry over from the original config unchanged
    adaptive_keys = (
        'adaptive_coefficients',
        'adaptive_baselines',
        'adaptive_mode',
        'adaptive_max_shift',
        'adaptive_coefficients2',
        'adaptive_power',
        'adaptive_band',
        'adaptive_sharpness',
    )

    # Aggregate outputs per process as numpy arrays
    aggregated: Dict[str, list] = {name: [] for name in process_order}

    for traj in calibration_trajectories:
        for process_name in process_order:
            if process_name not in traj:
                continue
            data = traj[process_name]
            out = data.get('outputs_mean')
            if out is None:
                out = data.get('outputs_sampled')
            if out is None:
                continue

            if isinstance(out, torch.Tensor):
                out_np = out.detach().cpu().numpy()
            else:
                out_np = np.asarray(out)

            if out_np.ndim == 1:
                out_np = out_np[:, None]
            aggregated[process_name].append(out_np)

    new_configs: Dict = {}
    for process_name in process_order:
        orig = dict(process_configs.get(process_name, {}))
        new_cfg: Dict = {}

        # Carry over adaptive keys unchanged
        for key in adaptive_keys:
            if key in orig:
                new_cfg[key] = orig[key]

        chunks = aggregated.get(process_name, [])
        if chunks:
            stacked = np.concatenate(chunks, axis=0)  # (N_total, d_t)
            mean_per_dim = stacked.mean(axis=0)
            std_per_dim = stacked.std(axis=0)
            new_cfg['shekel_center'] = [float(x) for x in mean_per_dim.tolist()]
            new_cfg['shekel_sigma'] = [float(x) for x in std_per_dim.tolist()]
        else:
            # No calibration data: fall back to base_target / scale if present,
            # else scalar defaults.
            base_target = orig.get('base_target', 0.0)
            if not isinstance(base_target, list):
                base_target = [float(base_target)]
            scale = orig.get('scale', 1.0)
            if not isinstance(scale, list):
                scale = [float(scale)] * len(base_target)
            new_cfg['shekel_center'] = [float(x) for x in base_target]
            # 'scale' in ReliabilityFunction is used as exp(-d^2/scale), which is
            # closer to a variance than a std; take sqrt as a coarse fallback.
            new_cfg['shekel_sigma'] = [float(np.sqrt(max(x, 1e-8))) for x in scale]

        new_configs[process_name] = new_cfg

    return new_configs
