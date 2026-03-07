"""
Styblinski-Tang Synthetic SCM Dataset.

Provides a configurable synthetic dataset based on the Styblinski-Tang function,
mapped onto the AZIMUTH SCM node-and-DAG framework. Complexity is controlled
through a single STConfig dataclass; the builder function build_st_scm()
produces SCMDataset instances automatically.

Six independent complexity axes:
  - Input dimensionality (n)
  - Chain depth (m)
  - Width profile (width_profile)
  - Environmental variables (me, env_mode, env_overlap)
  - Process noise (rho)
  - Output structure (p, output_overlap)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import numpy as np

from .scm import NodeSpec, SCMDataset


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class STConfig:
    """Configuration for the Styblinski-Tang SCM dataset.

    All complexity axes are controlled through this single object.
    The builder function ``build_st_scm(cfg)`` reads it and produces
    the correct NodeSpec list, noise samplers, and metadata.
    """

    # --- Dimensionality ---
    n: int = 6                          # Total input variables (must be >= m)
    m: int = 1                          # Number of cascaded ST stages

    # --- Width profile ---
    width_profile: str = 'uniform'      # Shape function name
    width_beta: float = 0.5             # Decay/growth rate for exp profiles
    width_mu: Optional[float] = None    # Peak stage for Gaussian profile
    width_s: float = 1.0                # Std dev for Gaussian profile

    # --- Environmental variables ---
    me: int = 1                         # Number of environmental variables
    env_mode: str = 'A'                 # Coupling mode: 'A', 'B', 'C', 'D'
    env_overlap: float = 0.0            # Fraction of shared dimension groups
    alpha: float = 1.0                  # Additive / shift env amplitude
    gamma: float = 0.2                  # Multiplicative env amplitude

    # --- Process noise ---
    rho: float = 0.3                    # Unified noise intensity [0, 1]
    sigma_max: float = 1.0              # Max Gaussian noise std at rho=1
    sigma_m_max: float = 0.3            # Max lognormal shape at rho=1
    lambda_max: float = 0.1             # Max Poisson jump rate at rho=1
    theta_jump: float = 1.0             # Exponential jump scale

    # --- Multi-output ---
    p: int = 1                          # Number of output nodes
    output_overlap: bool = False        # Overlapping output partitions

    # --- Miscellaneous ---
    carry_beta: float = 1.0             # Weight for carrying S_{k-1}
    x_domain: Tuple[float, float] = (-5.0, 5.0)  # Sampling domain for X_i
    e_min: float = -1.0                 # Environmental variable lower bound
    e_max: float = 1.0                  # Environmental variable upper bound


# ============================================================================
# Width-profile helpers
# ============================================================================

def _shape_fn(profile: str, m: int, beta: float = 0.5,
              mu: Optional[float] = None, s: float = 1.0):
    """Return a list of raw shape weights f(1), ..., f(m)."""
    if mu is None:
        mu = m / 2.0

    if profile == 'uniform':
        return [1.0] * m
    elif profile == 'exp_back':
        return [math.exp(beta * k) for k in range(1, m + 1)]
    elif profile == 'exp_front':
        return [math.exp(-beta * k) for k in range(1, m + 1)]
    elif profile == 'linear_back':
        return [float(k) for k in range(1, m + 1)]
    elif profile == 'linear_front':
        return [float(m + 1 - k) for k in range(1, m + 1)]
    elif profile == 'gaussian':
        return [math.exp(-(k - mu) ** 2 / (2 * s ** 2)) for k in range(1, m + 1)]
    else:
        raise ValueError(f"Unknown width_profile: {profile!r}")


def _compute_stage_widths(cfg: STConfig) -> List[int]:
    """Compute (n_1, ..., n_m) from the config using Eq. 3.

    Each stage gets at least 1 input; remaining n - m inputs are
    distributed according to the shape function, with largest-remainder
    rounding to ensure the sum equals n exactly.
    """
    n, m = cfg.n, cfg.m
    assert n >= m, f"n ({n}) must be >= m ({m})"

    if m == 1:
        return [n]

    raw = _shape_fn(cfg.width_profile, m, cfg.width_beta, cfg.width_mu, cfg.width_s)
    total = sum(raw)
    w = [r / total for r in raw]  # normalised weights

    extra = n - m  # inputs to distribute beyond the guaranteed 1 per stage
    floors = [math.floor(wk * extra) for wk in w]
    remainders = [wk * extra - f for wk, f in zip(w, floors)]

    # distribute correction terms to stages with largest fractional remainder
    deficit = extra - sum(floors)
    indices_by_remainder = sorted(range(m), key=lambda i: -remainders[i])
    corrections = [0] * m
    for i in range(deficit):
        corrections[indices_by_remainder[i]] = 1

    widths = [1 + floors[k] + corrections[k] for k in range(m)]
    assert sum(widths) == n, f"Width sum {sum(widths)} != n {n}"
    return widths


# ============================================================================
# Environment dimension-group assignment
# ============================================================================

def _assign_env_groups(n: int, me: int, overlap: float) -> List[List[int]]:
    """Assign dimension groups G_j for each environmental variable.

    With overlap=0, groups are disjoint.  With overlap>0, adjacent groups
    share a fraction of their dimensions.

    Returns a list of me lists, each containing 0-based dimension indices.
    """
    if me == 0:
        return []
    if me == 1:
        return [list(range(n))]

    # base group size (before overlap)
    base_size = max(1, n // me)
    groups: List[List[int]] = []

    for j in range(me):
        start = j * base_size
        end = min(start + base_size, n)
        dims = list(range(start, end))

        # add overlap with the next group's dimensions
        if overlap > 0 and j < me - 1:
            n_overlap = max(1, int(overlap * base_size))
            for d in range(end, min(end + n_overlap, n)):
                if d not in dims:
                    dims.append(d)

        groups.append(dims)

    return groups


# ============================================================================
# Multi-output input partitioning
# ============================================================================

def _partition_inputs(n: int, p: int, overlap: bool) -> List[List[int]]:
    """Partition n input dimensions into p subsets for multi-output.

    If overlap is False, partitions are disjoint.
    If overlap is True, adjacent partitions share boundary dimensions.
    """
    if p == 1:
        return [list(range(n))]

    base_size = max(1, n // p)
    partitions: List[List[int]] = []
    for k in range(p):
        start = k * base_size
        end = min(start + base_size, n) if k < p - 1 else n
        dims = list(range(start, end))

        if overlap and k > 0:
            # share one dimension with the previous partition
            boundary = start - 1
            if boundary >= 0 and boundary not in dims:
                dims.insert(0, boundary)

        partitions.append(dims)

    return partitions


# ============================================================================
# Expression builders
# ============================================================================

def _st_term(var: str) -> str:
    """Return the ST contribution for a single variable: x^4 - 16x^2 + 5x."""
    return f"({var}**4 - 16*{var}**2 + 5*{var})"


def _build_stage_expr(input_names: List[str], prev_stage: Optional[str],
                      carry_beta: float, stage_name: str,
                      env_shifts: Optional[dict] = None) -> str:
    """Build the SymPy expression string for a stage node.

    Parameters
    ----------
    input_names : variable names assigned to this stage
    prev_stage : name of the previous stage node (None for stage 1)
    carry_beta : weight for the carry-over from previous stage
    stage_name : name of this stage node (for the eps_ term)
    env_shifts : optional dict mapping input_name -> shift expression string
                 (for Mode C/D environment coupling)
    """
    terms = []
    for x in input_names:
        if env_shifts and x in env_shifts:
            shifted = f"({x} + {env_shifts[x]})"
            terms.append(_st_term(shifted))
        else:
            terms.append(_st_term(x))

    st_sum = " + ".join(terms)
    expr = f"0.5*({st_sum})"

    if prev_stage is not None:
        expr += f" + {carry_beta}*{prev_stage}"

    expr += f" + 0*eps_{stage_name}"
    return expr


def _build_output_expr(stage_name: str, env_names: List[str],
                       env_mode: str, alpha: float, gamma: float,
                       sigma_eff: float, sigma_m_eff: float,
                       output_name: str) -> str:
    """Build the output node expression implementing env mode + noise (Eq. 11).

    The noise source nodes (Z_ln, Eps_add, Jump) are parents, and their
    noise is already sampled by the SCM engine.  The expression here
    combines them with the environmental effect.
    """
    S = stage_name
    env_sum = " + ".join(env_names) if env_names else "0"

    # Lognormal: exp(-sigma_m^2/2 + sigma_m * eps_Z_ln)
    # We bake sigma_m_eff into the expression
    z_ln_expr = f"exp(-{sigma_m_eff**2/2.0} + {sigma_m_eff}*Z_ln_{output_name})"

    # Gaussian additive noise scaled by sigma_eff
    eps_add_expr = f"{sigma_eff}*Eps_add_{output_name}"

    # Jump is passed through directly (sampled at effective rate)
    jump_expr = f"Jump_{output_name}"

    if env_mode == 'A':
        base = f"{S} + {alpha}*({env_sum})"
    elif env_mode == 'B':
        base = f"(1 + {gamma}*({env_sum}))*{S}"
    elif env_mode == 'C':
        # shift already applied in stage expressions
        base = S
    elif env_mode == 'D':
        base = f"(1 + {gamma}*({env_sum}))*{S} + {alpha}*({env_sum})"
    else:
        raise ValueError(f"Unknown env_mode: {env_mode!r}")

    expr = f"{base}*{z_ln_expr} + {eps_add_expr} + {jump_expr} + 0*eps_{output_name}"
    return expr


# ============================================================================
# Builder
# ============================================================================

def build_st_scm(cfg: STConfig) -> SCMDataset:
    """Build a Styblinski-Tang SCM dataset from configuration.

    Returns an SCMDataset with all nodes, samplers, and metadata
    fully constructed from the STConfig object.
    """
    nodes: List[NodeSpec] = []
    singles = {}

    # Effective noise parameters
    sigma_eff = cfg.rho * cfg.sigma_max
    sigma_m_eff = cfg.rho * cfg.sigma_m_max
    lambda_eff = cfg.rho * cfg.lambda_max
    theta = cfg.theta_jump

    # ------------------------------------------------------------------
    # Step 1 — Width profile
    # ------------------------------------------------------------------
    partitions_out = _partition_inputs(cfg.n, cfg.p, cfg.output_overlap)

    # ------------------------------------------------------------------
    # Step 2 — Input nodes
    # ------------------------------------------------------------------
    x_names = [f"X_{i+1}" for i in range(cfg.n)]
    for xn in x_names:
        nodes.append(NodeSpec(xn, [], f"eps_{xn}"))
        lo, hi = cfg.x_domain
        singles[xn] = lambda rng, n, lo=lo, hi=hi: rng.uniform(lo, hi, n)

    # ------------------------------------------------------------------
    # Step 3 — Environmental nodes
    # ------------------------------------------------------------------
    e_names = [f"E_{j+1}" for j in range(cfg.me)]
    env_groups = _assign_env_groups(cfg.n, cfg.me, cfg.env_overlap)

    for en in e_names:
        nodes.append(NodeSpec(en, [], f"eps_{en}"))
        e_lo, e_hi = cfg.e_min, cfg.e_max
        singles[en] = lambda rng, n, lo=e_lo, hi=e_hi: rng.uniform(lo, hi, n)

    # Pre-compute env shift expressions per input dimension (Mode C/D)
    env_shift_map = {}  # x_name -> shift expression string
    if cfg.env_mode in ('C', 'D'):
        for j, en in enumerate(e_names):
            if j < len(env_groups):
                for dim_idx in env_groups[j]:
                    xn = x_names[dim_idx]
                    shift_term = f"{cfg.alpha}*{en}"
                    if xn in env_shift_map:
                        env_shift_map[xn] += f" + {shift_term}"
                    else:
                        env_shift_map[xn] = shift_term

    # ------------------------------------------------------------------
    # Steps 4-6 — Per-output chain: stages, noise sources, output node
    # ------------------------------------------------------------------
    all_stage_names = []
    noise_source_names = []
    output_names = []

    for out_idx in range(cfg.p):
        suffix = f"_{out_idx+1}" if cfg.p > 1 else ""
        out_name = f"Y{suffix}" if cfg.p > 1 else "Y"

        # Determine which input dimensions belong to this output
        out_dims = partitions_out[out_idx]
        out_x_names = [x_names[d] for d in out_dims]
        n_out = len(out_x_names)

        # Build a sub-config for width computation within this output's inputs
        sub_cfg = STConfig(n=n_out, m=cfg.m, width_profile=cfg.width_profile,
                           width_beta=cfg.width_beta, width_mu=cfg.width_mu,
                           width_s=cfg.width_s)
        widths = _compute_stage_widths(sub_cfg)

        # Stage chain for this output
        dim_cursor = 0
        prev_stage_name = None
        last_stage_name = None

        for k in range(cfg.m):
            stage_name = f"S_{k+1}{suffix}" if cfg.p > 1 else f"S_{k+1}"
            n_k = widths[k]
            stage_inputs = out_x_names[dim_cursor:dim_cursor + n_k]
            dim_cursor += n_k

            # Build parent list
            parents = list(stage_inputs)
            if prev_stage_name is not None:
                parents.append(prev_stage_name)

            # For Mode C/D, E_j nodes that affect this stage's inputs
            # must also be parents
            stage_env_parents = set()
            stage_env_shifts = {}
            if cfg.env_mode in ('C', 'D'):
                for xn in stage_inputs:
                    if xn in env_shift_map:
                        stage_env_shifts[xn] = env_shift_map[xn]
                        # extract E_j names from the shift expression
                        for en in e_names:
                            if en in env_shift_map[xn]:
                                stage_env_parents.add(en)

            for en in sorted(stage_env_parents):
                if en not in parents:
                    parents.append(en)

            expr = _build_stage_expr(
                stage_inputs, prev_stage_name, cfg.carry_beta,
                stage_name, stage_env_shifts if stage_env_shifts else None
            )

            nodes.append(NodeSpec(stage_name, parents, expr))
            singles[stage_name] = lambda rng, n: np.zeros(n)
            all_stage_names.append(stage_name)

            prev_stage_name = stage_name
            last_stage_name = stage_name

        # Noise source nodes (per output)
        z_ln_name = f"Z_ln{suffix}" if cfg.p > 1 else "Z_ln"
        eps_add_name = f"Eps_add{suffix}" if cfg.p > 1 else "Eps_add"
        jump_name = f"Jump{suffix}" if cfg.p > 1 else "Jump"

        # Rename in output expr to match node names
        z_ln_out = f"Z_ln_{out_name}"
        eps_add_out = f"Eps_add_{out_name}"
        jump_out = f"Jump_{out_name}"

        # Z_ln: standard normal, transformed in output expr
        nodes.append(NodeSpec(z_ln_name, [], f"eps_{z_ln_name}"))
        singles[z_ln_name] = lambda rng, n: rng.standard_normal(n)

        # Eps_add: standard normal, scaled in output expr
        nodes.append(NodeSpec(eps_add_name, [], f"eps_{eps_add_name}"))
        singles[eps_add_name] = lambda rng, n: rng.standard_normal(n)

        # Jump: Poisson-exponential compound sampler
        nodes.append(NodeSpec(jump_name, [], f"eps_{jump_name}"))
        _lam = lambda_eff
        _th = theta
        singles[jump_name] = _make_jump_sampler(_lam, _th)

        noise_source_names.extend([z_ln_name, eps_add_name, jump_name])

        # Output node
        # Parents: last stage, all E_j, noise sources
        out_parents = [last_stage_name]
        out_parents.extend(e_names)
        out_parents.extend([z_ln_name, eps_add_name, jump_name])

        # Build output expression — we need to use the actual node names
        # (not the aliased ones used in _build_output_expr)
        out_expr = _build_output_expr_direct(
            last_stage_name, e_names, cfg.env_mode,
            cfg.alpha, cfg.gamma,
            sigma_eff, sigma_m_eff,
            z_ln_name, eps_add_name, jump_name,
            out_name
        )

        nodes.append(NodeSpec(out_name, out_parents, out_expr))
        singles[out_name] = lambda rng, n: np.zeros(n)
        output_names.append(out_name)

    # ------------------------------------------------------------------
    # Step 7 — Assemble SCMDataset
    # ------------------------------------------------------------------
    target_labels = output_names
    input_labels = list(x_names)

    ds = SCMDataset(
        name="styblinski_tang_scm",
        description=(
            f"Styblinski-Tang SCM: n={cfg.n}, m={cfg.m}, "
            f"profile={cfg.width_profile}, me={cfg.me}, "
            f"env_mode={cfg.env_mode}, rho={cfg.rho}, p={cfg.p}"
        ),
        tags=["styblinski-tang", "synthetic", "scm"],
        specs=nodes,
        params={},
        singles=singles,
        groups=None,
        input_labels=input_labels,
        target_labels=target_labels,
    )

    ds.structural_noise_vars = list(e_names)
    ds.process_noise_vars = list(noise_source_names)

    return ds


# ============================================================================
# Internal helpers
# ============================================================================

def _make_jump_sampler(lambda_eff: float, theta: float):
    """Create a Poisson-exponential compound jump sampler."""
    def _sampler(rng, n, _lam=lambda_eff, _th=theta):
        if _lam <= 0:
            return np.zeros(n)
        result = np.zeros(n)
        for i in range(n):
            k = rng.poisson(_lam)
            if k > 0:
                result[i] = np.sum(rng.exponential(_th, k))
        return result
    return _sampler


def _build_output_expr_direct(stage_name: str, env_names: List[str],
                              env_mode: str, alpha: float, gamma: float,
                              sigma_eff: float, sigma_m_eff: float,
                              z_ln_name: str, eps_add_name: str,
                              jump_name: str, output_name: str) -> str:
    """Build output expression using actual node names as SymPy symbols."""
    S = stage_name
    env_sum = " + ".join(env_names) if env_names else "0"

    # Lognormal multiplicative noise
    sm2 = sigma_m_eff ** 2 / 2.0
    z_ln_expr = f"exp(-{sm2} + {sigma_m_eff}*{z_ln_name})"

    # Gaussian additive noise
    eps_expr = f"{sigma_eff}*{eps_add_name}"

    if env_mode == 'A':
        base = f"{S} + {alpha}*({env_sum})"
    elif env_mode == 'B':
        base = f"(1 + {gamma}*({env_sum}))*{S}"
    elif env_mode == 'C':
        base = S
    elif env_mode == 'D':
        base = f"(1 + {gamma}*({env_sum}))*{S} + {alpha}*({env_sum})"
    else:
        raise ValueError(f"Unknown env_mode: {env_mode!r}")

    return (f"({base})*{z_ln_expr} + {eps_expr} + {jump_name} "
            f"+ 0*eps_{output_name}")
