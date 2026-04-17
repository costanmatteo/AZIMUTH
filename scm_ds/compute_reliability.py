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
from typing import Dict, List, Tuple, Optional, Union

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
                 device: str = 'cpu',
                 reliability_formula: str = 'gaussian',
                 shekel_sharpness: float = 1.0):
        """
        Args:
            process_configs: Process-specific targets and weights.
                            If None, uses default PROCESS_CONFIGS.
            process_order: Order of processes for dependency resolution.
                          If None, uses default PROCESS_ORDER.
            device: Torch device for computations.
            reliability_formula: 'gaussian' (default, weighted Q average) or
                                 'shekel' (global Shekel function, eq. 2.6).
            shekel_sharpness: Global sharpness hyperparameter s > 0 used to
                              calibrate Shekel width coefficients d_t^k = s / Var[o_t^k].
                              Only used when reliability_formula='shekel'.
        """
        self.process_configs = process_configs or PROCESS_CONFIGS
        self.process_order = process_order or PROCESS_ORDER
        self.device = device
        self.reliability_formula = reliability_formula
        self.shekel_sharpness = shekel_sharpness
        self._shekel_widths: Optional[Dict[str, torch.Tensor]] = None

    def calibrate_shekel_widths(self, trajectories: List[Dict]) -> None:
        """
        Calibrate Shekel width coefficients d_t^k = shekel_sharpness / Var[o_t^k].

        Must be called once before compute_reliability() when
        reliability_formula='shekel'.

        Args:
            trajectories: List of trajectory dicts (same format as
                compute_reliability input). Each trajectory maps process_name
                to {'outputs_mean': tensor, 'outputs_sampled': tensor, ...}.

        Stores:
            self._shekel_widths: Dict[process_name, tensor(output_dim)]
        """
        # Collect all outputs per process across trajectories
        collected: Dict[str, list] = {}
        for traj in trajectories:
            for process_name, data in traj.items():
                if process_name not in collected:
                    collected[process_name] = []
                if 'outputs_sampled' in data:
                    out = data['outputs_sampled']
                else:
                    out = data['outputs_mean']
                if isinstance(out, np.ndarray):
                    out = torch.tensor(out, dtype=torch.float32)
                if out.dim() == 1:
                    out = out.unsqueeze(0)
                collected[process_name].append(out)

        self._shekel_widths = {}
        for process_name in self.process_order:
            if process_name not in collected:
                continue
            # Stack: (N_cal, output_dim) — squeeze batch dim if each entry is (1, output_dim)
            stacked = torch.cat(collected[process_name], dim=0)  # (N_cal, output_dim)
            var = stacked.var(dim=0)  # (output_dim,)
            # Clamp variance to avoid division by zero
            var = torch.clamp(var, min=1e-8)
            self._shekel_widths[process_name] = (self.shekel_sharpness / var).to(self.device)

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
            return_quality_scores: If True, also return per-process scores.
                For 'gaussian': per-process Q_i scores (normalised quality).
                For 'shekel': per-process partial sums (unnormalised
                    sum_k d_t^k * (o_t^k - zeta*_t^k)^2).
            use_sampled_outputs: If True, use 'outputs_sampled' if available,
                                otherwise use 'outputs_mean'.

        Returns:
            F: Reliability score (batch,) or scalar
            quality_scores: Dict of per-process scores (if return_quality_scores=True)
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

        # Branch on reliability formula
        if self.reliability_formula == 'shekel':
            return self._compute_shekel(outputs, return_quality_scores)

        # ── Gaussian path — chain variant for adaptive targets ───────────
        # adaptive_targets stores τ_j per-dim for every processed upstream so
        # that downstream processes can use it as baseline (chain variant).
        adaptive_targets: Dict[str, torch.Tensor] = {}
        quality_scores = {}

        batch_size = next(iter(outputs.values())).shape[0]

        for process_name in self.process_order:
            if process_name not in outputs:
                continue

            config = self.process_configs.get(process_name, {})
            output = outputs[process_name]

            # Compute adaptive target τ_i.
            target = self._compute_adaptive_target(
                process_name, outputs, adaptive_targets, config
            )

            # Normalise to (batch, output_dim_i) before storing so downstream
            # can safely do Y_i - τ_i per-dim regardless of process position.
            if target.dim() == 1:                                    # (output_dim,)
                target_stored = target.unsqueeze(0).expand(batch_size, -1)
            elif target.shape[0] == 1 and batch_size > 1:            # (1, output_dim)
                target_stored = target.expand(batch_size, -1)
            else:                                                    # (batch, output_dim)
                target_stored = target
            adaptive_targets[process_name] = target_stored

            scales = config.get('scale', 1.0)
            if not isinstance(scales, list):
                scales = [scales]
            scale_t = torch.tensor(scales, dtype=torch.float32, device=self.device)  # (output_dim,)

            # Broadcast of target against output is correct: target is either
            # (output_dim,) for the first process or (batch, output_dim) for
            # adaptive ones. (batch, output_dim) - (output_dim,) also broadcasts.
            per_dim_q = torch.exp(-((output - target) ** 2) / scale_t)  # (batch, output_dim)
            quality = per_dim_q.mean(dim=-1)                             # (batch,)

            quality_scores[process_name] = quality

        # Compute weighted average reliability
        F = self._compute_weighted_average(quality_scores)

        if return_quality_scores:
            return F, quality_scores
        return F

    def _compute_shekel(self,
                        outputs: Dict[str, torch.Tensor],
                        return_quality_scores: bool) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Shekel reliability path (eq. 2.6):
            F(o) = 1 / (1 + sum_t sum_k  d_t^k * (o_t^k - zeta*_t^k)^2)

        Args:
            outputs: Dict mapping process_name to tensor (batch, output_dim).
            return_quality_scores: If True, also return per-process partial sums.

        Returns:
            F: (batch,) tensor.
            partial_sums: Dict of per-process partial sums (if return_quality_scores).
        """
        if self._shekel_widths is None:
            raise RuntimeError(
                "Shekel widths not calibrated. Call calibrate_shekel_widths() "
                "before compute_reliability() when reliability_formula='shekel'."
            )

        # Determine batch size from first available output
        batch_size = next(iter(outputs.values())).shape[0]
        total_sum = torch.zeros(batch_size, device=self.device)
        partial_sums = {}
        # adaptive_targets stores τ_j per-dim for every processed upstream
        # (chain variant: τ_j is used as baseline by downstream processes).
        adaptive_targets: Dict[str, torch.Tensor] = {}

        for process_name in self.process_order:
            if process_name not in outputs:
                continue

            config = self.process_configs.get(process_name, {})
            output = outputs[process_name]  # (batch, output_dim)

            # Adaptive target — chain variant.
            target = self._compute_adaptive_target(
                process_name, outputs, adaptive_targets, config
            )

            if target.dim() == 1:
                target_stored = target.unsqueeze(0).expand(batch_size, -1)
            elif target.shape[0] == 1 and batch_size > 1:
                target_stored = target.expand(batch_size, -1)
            else:
                target_stored = target
            adaptive_targets[process_name] = target_stored

            # Width coefficients d_t^k for this process
            d = self._shekel_widths[process_name]  # (output_dim,)

            # Per-dimension squared deviation weighted by d_t^k
            diff_sq = (output - target) ** 2  # (batch, output_dim)
            weighted = d * diff_sq             # (batch, output_dim)
            proc_sum = weighted.sum(dim=-1)    # (batch,)

            total_sum = total_sum + proc_sum
            partial_sums[process_name] = proc_sum

        F = 1.0 / (1.0 + total_sum)  # (batch,)

        if return_quality_scores:
            return F, partial_sums
        return F

    def _compute_adaptive_target(self,
                                 process_name: str,
                                 outputs: Dict[str, torch.Tensor],
                                 adaptive_targets: Dict[str, torch.Tensor],
                                 config: Dict) -> torch.Tensor:
        """
        Compute adaptive target τ_i for a process using the chain variant:

            τ_i[k] = ζ⁽⁰⁾_i[k] + Σ_{j<i} coeff_j · mean_q( f(Y_j[q] − τ_j[q]) )

        The baseline for each upstream j is τ_j (its own adaptive target
        computed earlier in the forward pass), NOT the static ζ⁽⁰⁾_j. The
        'adaptive_baselines' field in config is therefore ignored.

        Per-dim semantics: delta is computed per-dimension on the upstream
        output (Y_j − τ_j), f is applied per-dim, then collapsed to a scalar
        shift per sample via mean over upstream dims, and added to the
        per-dim base_target of i. τ_i keeps shape (batch, output_dim_i).

        Returns:
            - (output_dim_i,) when no adaptive coefficients are configured
              (first process in the chain).
            - (batch, output_dim_i) when adaptive coefficients are present.
        """
        base_target = config.get('base_target', 0.0)

        if isinstance(base_target, list):
            base_target_t = torch.tensor(base_target, dtype=torch.float32, device=self.device)
        else:
            base_target_t = torch.tensor([base_target], dtype=torch.float32, device=self.device)

        adaptive_coeffs = config.get('adaptive_coefficients', {})

        if not adaptive_coeffs:
            return base_target_t  # (output_dim,) — downstream broadcasts

        mode = config.get('adaptive_mode', 'linear')
        adaptive_coefficients2 = config.get('adaptive_coefficients2', {})
        adaptive_power = config.get('adaptive_power', {})
        adaptive_sharpness = config.get('adaptive_sharpness', {})
        adaptive_band = config.get('adaptive_band', {})
        adaptive_max_shift = config.get('adaptive_max_shift', {})

        target = base_target_t.unsqueeze(0)  # (1, output_dim_i) — broadcasts over batch

        for upstream_name, coeff in adaptive_coeffs.items():
            if upstream_name not in outputs:
                continue
            if upstream_name not in adaptive_targets:
                # process_order should guarantee upstream τ is already computed;
                # a miss indicates a topology/order bug — skip defensively.
                continue

            upstream_out = outputs[upstream_name]        # (batch, m_upstream)
            tau_j = adaptive_targets[upstream_name]      # (batch, m_upstream), pre-normalised

            delta = upstream_out - tau_j                 # (batch, m_upstream) per-dim

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
            # Collapse upstream dims → scalar shift per sample. Mean (not sum)
            # keeps the shift magnitude independent of upstream dimensionality.
            shift = adjustment.mean(dim=-1, keepdim=True)  # (batch, 1)
            target = target + shift  # (1, output_dim_i) + (batch, 1) → (batch, output_dim_i)

        # Ensure (batch, output_dim_i). If no upstream contributed, target is
        # still (1, output_dim_i); expand to match any downstream batch later.
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
                       device: str = 'cpu',
                       reliability_formula: str = 'gaussian',
                       shekel_sharpness: float = 1.0) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
    """
    Convenience function to compute reliability F.

    Args:
        trajectory: Trajectory dict (see ReliabilityFunction.compute_reliability).
        return_quality_scores: If True, also return per-process scores.
        device: Torch device.
        reliability_formula: 'gaussian' or 'shekel'.
        shekel_sharpness: Global sharpness s for Shekel width calibration.
    """
    rf = ReliabilityFunction(device=device,
                             reliability_formula=reliability_formula,
                             shekel_sharpness=shekel_sharpness)
    return rf.compute_reliability(trajectory, return_quality_scores=return_quality_scores)
