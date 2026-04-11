"""
Sigmoid-Cubic SCM Dataset — Fully synthetic, configurable SCM datasets.

Builds SCMDataset instances whose structural equations use a sigmoid-cubic
transfer function  f(x) = 1/(1+exp(-(x³-3x)))  arranged in a compositional
chain-of-stages DAG.  Each stage applies f() to the sum of its fresh inputs
plus the rescaled carry from the previous stage.  After each intermediate
variable is computed, it is rescaled back to [-2, 2] using the empirical
min/max of the layer.  Complexity is controlled through a single STConfig
dataclass.

The module provides:
    - STConfig: dataclass holding all configuration parameters.
    - build_st_scm: builder function that turns an STConfig into an SCMDataset.
    - Five pre-built instances (ds_scm_st_minimal, ..., ds_scm_st_multi).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .scm import NodeSpec, SCMDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class STConfig:
    """All parameters for a Styblinski-Tang SCM dataset."""

    # --- input / depth ---
    n: int = 4                          # total input variables
    m: int = 1                          # number of cascaded ST stages

    # --- width profile ---
    width_profile: str = "uniform"      # uniform|exp_back|exp_front|linear_back|linear_front|gaussian
    width_beta: float = 0.5             # rate for exp profiles
    width_mu: Optional[float] = None    # peak stage for gaussian (default m/2)
    width_s: float = 1.0                # std for gaussian profile

    # --- environment ---
    me: int = 0                         # number of environmental variables
    env_mode: str = "A"                 # A|B|C|D
    env_overlap: float = 0.0            # overlap fraction [0,1]
    alpha: float = 0.3                  # additive / input-shift amplitude
    gamma: float = 0.3                  # multiplicative amplitude

    # --- noise ---
    rho: float = 0.0                    # unified noise intensity [0,1]
    sigma_max: float = 0.20            # max gaussian noise std
    sigma_m_max: float = 0.20          # max lognormal shape
    lambda_max: float = 0.50           # max Poisson jump rate
    theta_jump: float = 0.10           # exponential jump scale

    # --- multi-output ---
    p: int = 1                          # number of output nodes
    output_overlap: float = 0.0         # overlap fraction [0,1] between output partitions

    # --- carry ---
    carry_beta: float = 1.0            # weight of S_{k-1} in stage k

    # --- domains ---
    x_domain: Tuple[float, float] = (-2.0, 2.0)
    e_domain: Tuple[float, float] = (-1.0, 1.0)
    action_domain: Tuple[float, float] = (-2.0, 2.0)  # controller action range for calibration

    # --- calibration ---
    cal_n: int = 2000               # calibration sample size
    cal_seed: int = 99              # calibration RNG seed
    cal_percentile: float = 10.0    # percentile for base_target (lower = harder)
    cal_width_factor: float = 1.0   # scale = (std * width_factor)^2


# ---------------------------------------------------------------------------
# Width profile helpers
# ---------------------------------------------------------------------------

_PROFILES = {"uniform", "exp_back", "exp_front", "linear_back", "linear_front", "gaussian"}


def _compute_width(cfg: STConfig, n_sub: int, m: int) -> List[int]:
    """Return list of length *m* summing to *n_sub*."""
    if m <= 0:
        return []
    if n_sub == 0:
        return [0] * m
    if m > n_sub:
        # first n_sub stages each get 1 input, remaining get 0
        return [1] * n_sub + [0] * (m - n_sub)

    profile = cfg.width_profile
    if profile not in _PROFILES:
        raise ValueError(f"Unknown width_profile '{profile}'. Choose from {_PROFILES}")

    # shape function values
    ks = list(range(1, m + 1))
    if profile == "uniform":
        f = [1.0] * m
    elif profile == "exp_back":
        f = [math.exp(cfg.width_beta * k) for k in ks]
    elif profile == "exp_front":
        f = [math.exp(-cfg.width_beta * k) for k in ks]
    elif profile == "linear_back":
        f = [float(k) for k in ks]
    elif profile == "linear_front":
        f = [float(m + 1 - k) for k in ks]
    elif profile == "gaussian":
        mu = cfg.width_mu if cfg.width_mu is not None else m / 2.0
        s = cfg.width_s
        f = [math.exp(-((k - mu) ** 2) / (2 * s * s)) for k in ks]
    else:
        raise ValueError(f"Unsupported profile: {profile}")

    total_f = sum(f)
    remaining = n_sub - m  # each stage gets at least 1
    raw = [f_k / total_f * remaining for f_k in f]
    counts = [1 + int(r) for r in raw]

    # distribute remainders
    deficit = n_sub - sum(counts)
    if deficit != 0:
        fracs = [(raw[i] - int(raw[i]), i) for i in range(m)]
        fracs.sort(key=lambda t: -t[0])
        for j in range(abs(deficit)):
            counts[fracs[j][1]] += 1 if deficit > 0 else -1

    assert sum(counts) == n_sub, f"Width profile bug: {sum(counts)} != {n_sub}"
    return counts


# ---------------------------------------------------------------------------
# Environmental group assignment
# ---------------------------------------------------------------------------

def _assign_env_groups(n: int, me: int, overlap: float) -> List[List[int]]:
    """Return list of *me* groups, each a list of input indices."""
    if me == 0:
        return []

    # round-robin base assignment
    groups: List[List[int]] = [[] for _ in range(me)]
    for i in range(n):
        groups[i % me].append(i)

    if overlap <= 0.0 or me < 2:
        return groups

    # add boundary overlap: share a fraction of inputs between adjacent groups
    for j in range(me - 1):
        g_curr = groups[j]
        g_next = groups[j + 1]
        n_share = max(1, int(overlap * min(len(g_curr), len(g_next))))
        # share the last n_share of g_curr into g_next (and vice versa first of g_next)
        for idx in g_curr[-n_share:]:
            if idx not in g_next:
                g_next.append(idx)
        for idx in g_next[:n_share]:
            if idx not in g_curr:
                g_curr.append(idx)

    return groups


# ---------------------------------------------------------------------------
# Expression builders
# ---------------------------------------------------------------------------

def _st_term(var: str) -> str:
    """Sigmoid-cubic transfer: 1/(1+exp(-(x³-3x))).  Bounded in (0, 1)."""
    return f"1/(1 + exp(-({var}**3 - 3*{var})))"


def _build_stage_expr(
    input_names: List[str],
    prev_stage: Optional[str],
    carry_beta: float,
    env_shifts: Optional[dict],  # {input_name: "alpha*(E1 + E2)" or None}
    eps_name: str,
) -> str:
    """Build the SymPy expression string for one stage node.

    Compositional structure: the sigmoid-cubic f() is applied to the
    **sum** of all inputs (fresh + rescaled carry from previous stage):

        S_1 = f(X_1 + X_2)                     (no carry)
        S_2 = f(rescale(S_1) + X_3)            (carry + fresh)
        S_3 = f(rescale(S_2))                   (carry only, no fresh)

    The rescaling to [-2, 2] is applied externally after each stage
    (see _rescaling_forward in build_st_scm).  carry_beta is unused
    in this formulation — the carry enters as a direct addend.
    """
    # Collect all addends: fresh inputs (with optional env shifts) + carry
    addends = []
    for x in input_names:
        if env_shifts and x in env_shifts:
            addends.append(f"({x} + {env_shifts[x]})")
        else:
            addends.append(x)

    if prev_stage is not None:
        addends.append(prev_stage)

    if len(addends) > 0:
        inner = " + ".join(addends)
        expr = _st_term(inner)
    else:
        expr = "0"

    # zero-coefficient noise term (required by SCM engine)
    expr = f"{expr} + 0*{eps_name}"
    return expr


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_st_scm(cfg: STConfig, dag_image_dir: Optional[str] = None) -> SCMDataset:
    """Build an SCMDataset from an STConfig.

    Parameters
    ----------
    cfg : STConfig
        Configuration dataclass.
    dag_image_dir : str or None
        If given, save a DAG visualisation PNG inside this directory.
    """

    # Validate
    if cfg.p < 1:
        raise ValueError("p must be >= 1")
    if cfg.p > cfg.n:
        raise ValueError(f"p ({cfg.p}) must be <= n ({cfg.n})")

    specs: List[NodeSpec] = []
    singles: dict = {}

    x_lo, x_hi = cfg.x_domain
    e_lo, e_hi = cfg.e_domain

    # ── STEP 1: Partition inputs for multi-output ──────────────────────
    all_input_names = [f"X_{i+1}" for i in range(cfg.n)]

    if cfg.p == 1:
        output_partitions = [list(range(cfg.n))]
    else:
        base_size = cfg.n // cfg.p
        remainder = cfg.n % cfg.p
        partitions: List[List[int]] = []
        idx = 0
        for oi in range(cfg.p):
            size = base_size + (1 if oi < remainder else 0)
            partitions.append(list(range(idx, idx + size)))
            idx += size
        if cfg.output_overlap > 0.0 and cfg.p > 1:
            for oi in range(cfg.p - 1):
                p_curr = partitions[oi]
                p_next = partitions[oi + 1]
                n_share = max(1, int(cfg.output_overlap * min(len(p_curr), len(p_next))))
                for idx in p_curr[-n_share:]:
                    if idx not in p_next:
                        p_next.append(idx)
                for idx in p_next[:n_share]:
                    if idx not in p_curr:
                        p_curr.append(idx)
        output_partitions = partitions

    # ── STEP 2: Input node specs ──────────────────────────────────────
    for i in range(cfg.n):
        name = all_input_names[i]
        specs.append(NodeSpec(name, [], f"eps_{name}"))
        singles[name] = lambda rng, n_, lo=x_lo, hi=x_hi: rng.uniform(lo, hi, size=n_)

    # ── STEP 3: Environmental node specs ──────────────────────────────
    env_names = [f"E_{j+1}" for j in range(cfg.me)]
    for j in range(cfg.me):
        name = env_names[j]
        specs.append(NodeSpec(name, [], f"eps_{name}"))
        singles[name] = lambda rng, n_, lo=e_lo, hi=e_hi: rng.uniform(lo, hi, size=n_)

    # Environmental groups (indices into all_input_names)
    env_groups = _assign_env_groups(cfg.n, cfg.me, cfg.env_overlap)

    # Build lookup: input_index -> list of env var names affecting it
    input_to_envs: dict = {}
    for j, grp in enumerate(env_groups):
        for idx in grp:
            input_to_envs.setdefault(idx, []).append(env_names[j])

    # ── STEP 4-8: Per-output chain ────────────────────────────────────
    all_stage_names: List[str] = []
    output_names: List[str] = []
    noise_node_names_zln: List[str] = []
    noise_node_names_eps: List[str] = []
    noise_node_names_jump: List[str] = []

    sigma_rho = cfg.rho * cfg.sigma_max
    sigma_m_rho = cfg.rho * cfg.sigma_m_max
    lambda_rho = cfg.rho * cfg.lambda_max

    for oi, partition in enumerate(output_partitions):
        suffix = "" if cfg.p == 1 else f"_{oi+1}"
        n_sub = len(partition)
        sub_input_names = [all_input_names[i] for i in partition]

        # Width profile for this chain
        m = cfg.m
        widths = _compute_width(cfg, n_sub, m)

        # Assign inputs to stages (contiguous slices)
        stage_inputs: List[List[str]] = []
        idx = 0
        for k in range(m):
            if idx < len(sub_input_names):
                stage_inputs.append(sub_input_names[idx: idx + widths[k]])
                idx += widths[k]
            else:
                stage_inputs.append([])   # no fresh input — carry only

        # ── Stage nodes ───────────────────────────────────────────
        prev_stage_name: Optional[str] = None
        for k in range(m):
            stage_name = f"S_{k+1}{suffix}"
            eps_name = f"eps_{stage_name}"

            # Determine parents
            parents = list(stage_inputs[k])
            if prev_stage_name is not None:
                parents.append(prev_stage_name)

            # Environment shifts for modes C and D
            env_shifts: Optional[dict] = None
            if cfg.env_mode in ("C", "D") and cfg.me > 0:
                env_shifts = {}
                for x_name in stage_inputs[k]:
                    x_idx = all_input_names.index(x_name)
                    if x_idx in input_to_envs:
                        envs = input_to_envs[x_idx]
                        shift_expr = " + ".join(envs)
                        if cfg.env_mode == "D":
                            env_shifts[x_name] = f"{cfg.alpha}*({shift_expr})"
                        else:
                            env_shifts[x_name] = f"{cfg.alpha}*({shift_expr})"
                        # Add env vars as parents
                        for ev in envs:
                            if ev not in parents:
                                parents.append(ev)

            expr = _build_stage_expr(
                stage_inputs[k], prev_stage_name, cfg.carry_beta, env_shifts, eps_name
            )
            specs.append(NodeSpec(stage_name, parents, expr))
            singles[stage_name] = lambda rng, n_: np.zeros(n_)

            all_stage_names.append(stage_name)
            prev_stage_name = stage_name

        last_stage = prev_stage_name

        # ── Noise nodes (per output chain) ────────────────────────
        zln_name = f"Z_ln{suffix}"
        eps_add_name = f"Eps_add{suffix}"
        jump_name = f"Jump{suffix}"

        noise_node_names_zln.append(zln_name)
        noise_node_names_eps.append(eps_add_name)
        noise_node_names_jump.append(jump_name)

        # Z_ln: lognormal transform (sampling logic in the sampler)
        specs.append(NodeSpec(zln_name, [], f"eps_{zln_name}"))
        sm = sigma_m_rho  # capture
        def _make_zln_sampler(sm_val):
            def _sampler(rng, n_):
                if sm_val <= 0:
                    return np.ones(n_)
                eta = rng.standard_normal(n_)
                return np.exp(-sm_val**2 / 2.0 + sm_val * eta)
            return _sampler
        singles[zln_name] = _make_zln_sampler(sm)

        # Eps_add: Gaussian additive (sampling logic in the sampler)
        specs.append(NodeSpec(eps_add_name, [], f"eps_{eps_add_name}"))
        sr = sigma_rho  # capture
        def _make_eps_sampler(sr_val):
            def _sampler(rng, n_):
                if sr_val <= 0:
                    return np.zeros(n_)
                return sr_val * rng.standard_normal(n_)
            return _sampler
        singles[eps_add_name] = _make_eps_sampler(sr)

        # Jump: compound Poisson-exponential (sampling in noise lambda)
        specs.append(NodeSpec(jump_name, [], f"eps_{jump_name}"))
        lam_rho = lambda_rho   # capture
        theta_j = cfg.theta_jump
        def _make_jump_sampler(lam, theta):
            def _sampler(rng, n_):
                if lam <= 0:
                    return np.zeros(n_)
                counts = rng.poisson(lam, size=n_)
                out = np.zeros(n_)
                for i in range(n_):
                    if counts[i] > 0:
                        out[i] = np.sum(rng.exponential(theta, size=counts[i]))
                return out
            return _sampler
        singles[jump_name] = _make_jump_sampler(lam_rho, theta_j)

        # ── Output node ───────────────────────────────────────────
        y_name = f"Y{suffix}"
        output_names.append(y_name)

        y_parents = [last_stage]
        env_sum = " + ".join(env_names) if env_names else "0"

        if cfg.env_mode == "A":
            # Y = S_m + alpha * sum(E_j)  (additive shift)
            if cfg.me > 0:
                y_clean = f"{last_stage} + {cfg.alpha}*({env_sum})"
                y_parents.extend(env_names)
            else:
                y_clean = last_stage
        elif cfg.env_mode == "B":
            # Y = (1 + gamma * sum(E_j)) * S_m
            if cfg.me > 0:
                y_clean = f"(1 + {cfg.gamma}*({env_sum}))*{last_stage}"
                y_parents.extend(env_names)
            else:
                y_clean = last_stage
        elif cfg.env_mode == "C":
            # shifts baked into stages; Y_clean = S_m
            y_clean = last_stage
        elif cfg.env_mode == "D":
            # combined: shift in stages, mult + add at output
            if cfg.me > 0:
                y_clean = f"(1 + {cfg.gamma}*({env_sum}))*{last_stage} + {cfg.alpha}*({env_sum})"
                y_parents.extend(env_names)
            else:
                y_clean = last_stage
        else:
            raise ValueError(f"Unknown env_mode '{cfg.env_mode}'")

        # Deduplicate parents
        seen = set()
        unique_parents = []
        for p in y_parents:
            if p not in seen:
                unique_parents.append(p)
                seen.add(p)
        y_parents = unique_parents

        # Add noise nodes as parents
        y_parents.extend([zln_name, eps_add_name, jump_name])

        # Y = Y_clean * Z_ln + Eps_add + Jump
        y_expr = f"({y_clean})*{zln_name} + {eps_add_name} + {jump_name} + 0*eps_{y_name}"
        specs.append(NodeSpec(y_name, y_parents, y_expr))
        singles[y_name] = lambda rng, n_: np.zeros(n_)

    # ── STEP 9: Assemble SCMDataset ───────────────────────────────────
    # Build description
    desc = (
        f"Sigmoid-cubic SCM: n={cfg.n}, m={cfg.m}, p={cfg.p}, "
        f"env_mode={cfg.env_mode}, me={cfg.me}, rho={cfg.rho}"
    )

    ds = SCMDataset(
        name=f"st_n{cfg.n}_m{cfg.m}_p{cfg.p}_{cfg.env_mode}",
        description=desc,
        tags=["sigmoid-cubic", "synthetic"],
        specs=specs,
        params={},
        singles=singles,
        groups=None,
        input_labels=list(all_input_names),
        target_labels=output_names,
    )

    # Noise classification
    ds.structural_noise_vars = list(env_names)
    all_noise_nodes = []
    for name_list in [noise_node_names_zln, noise_node_names_eps, noise_node_names_jump]:
        all_noise_nodes.extend(name_list)
    ds.process_noise_vars = all_noise_nodes

    # ── STEP 9b: Install per-layer rescaling to [-2, 2] ──────────────
    # After each intermediate stage node S_k is computed, rescale it
    # to [-2, 2] using the empirical min/max of that layer's values.
    # This is NOT applied to the final output Y.
    _stage_set = frozenset(all_stage_names)
    _scm_ref = ds.scm

    _original_forward = _scm_ref.forward

    def _rescaling_forward(context, eps_draws):
        for v in _scm_ref.order:
            parents = _scm_ref.specs[v].parents
            args = [context[p] for p in parents] + [eps_draws[v]]
            context[v] = _scm_ref._fns[v](*args)
            if v in _stage_set:
                vals = context[v]
                v_min = float(vals.min())
                v_max = float(vals.max())
                if v_max > v_min:
                    context[v] = -2.0 + 4.0 * (vals - v_min) / (v_max - v_min)
        return context

    _scm_ref.forward = _rescaling_forward

    # ── STEP 10: Calibrate process_configs ────────────────────────────
    _calibrate(ds, cfg, all_stage_names, output_names, output_partitions)

    # ── STEP 11: Save DAG image (optional) ────────────────────────────
    if dag_image_dir is not None:
        import os
        os.makedirs(dag_image_dir, exist_ok=True)
        filepath = os.path.join(dag_image_dir, ds.meta["name"])
        try:
            saved = ds.save_dag_image(filepath)
            print(f"DAG image saved: {saved}")
        except Exception as e:
            print(f"Warning: could not save DAG image: {e}")

    return ds


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _calibrate(
    ds: SCMDataset,
    cfg: STConfig,
    all_stage_names: List[str],
    output_names: List[str],
    output_partitions: List[List[int]],
) -> None:
    """Calibrate process_configs, process_order on *ds* using a rho=0 reference sample.

    When cfg.action_domain is set, calibration samples controllable inputs
    (X_i) from action_domain instead of x_domain so that base_target
    reflects what the controller can actually reach.
    """

    # ── Build a zero-noise clone for the reference sample ─────────
    original_singles = ds.noise_model.singles.copy()
    modified = {}

    # Determine calibration range for controllable inputs
    cal_lo, cal_hi = cfg.action_domain

    input_names_set = set(ds.input_labels)

    for var_name, fn in original_singles.items():
        if var_name in ds.process_noise_vars:
            # Zero out process noise
            if var_name.startswith("Z_ln"):
                # lognormal identity: return 1.0
                modified[var_name] = lambda rng, n_: np.ones(n_)
            else:
                modified[var_name] = lambda rng, n_: np.zeros(n_)
        elif var_name in input_names_set:
            # Sample controllable inputs from action_domain during calibration
            modified[var_name] = (
                lambda rng, n_, lo=cal_lo, hi=cal_hi: rng.uniform(lo, hi, size=n_)
            )
        else:
            modified[var_name] = fn
    ds.noise_model.singles = modified
    try:
        cal_df = ds.sample(n=cfg.cal_n, seed=cfg.cal_seed)
    finally:
        ds.noise_model.singles = original_singles

    # ── Identify per-chain stage sequences ────────────────────────
    # For p=1: stages are S_1, S_2, ..., S_m  (no suffix)
    # For p>1: chain i has S_1_i, S_2_i, ..., S_m_i
    chains: List[List[str]] = []  # each chain: list of stage names in order
    for oi in range(cfg.p):
        suffix = "" if cfg.p == 1 else f"_{oi+1}"
        n_sub = len(output_partitions[oi])
        m = min(cfg.m, n_sub)
        chain = [f"S_{k+1}{suffix}" for k in range(m)]
        chains.append(chain)

    # ── Build process_configs and process_order ───────────────────
    process_configs: Dict[str, dict] = {}
    process_order: List[str] = []
    pct = cfg.cal_percentile
    wf = cfg.cal_width_factor

    for oi, (chain, y_name) in enumerate(zip(chains, output_names)):
        # Stages first
        for ki, stage_name in enumerate(chain):
            vals = cal_df[stage_name].values
            tau = float(np.percentile(vals, pct))
            std = float(np.std(vals))
            scale = (std * wf) ** 2 if std > 0 else 1.0

            entry: dict = {
                "base_target": tau,
                "scale": scale,
                "weight": 1.0,
            }

            process_configs[stage_name] = entry
            process_order.append(stage_name)

        # Output node
        y_vals = cal_df[y_name].values
        y_tau = float(np.percentile(y_vals, pct))
        y_std = float(np.std(y_vals))
        y_scale = (y_std * wf) ** 2 if y_std > 0 else 1.0

        y_entry: dict = {
            "base_target": y_tau,
            "scale": y_scale,
            "weight": 1.0,
        }

        process_configs[y_name] = y_entry
        process_order.append(y_name)

    ds.process_configs = process_configs
    ds.process_order = process_order

    # ── Find calibration row closest to all base_targets ──────────
    # This row will be used as the single target trajectory so that
    # F* ≈ 1 by construction.
    all_nodes = list(process_configs.keys())
    scores = np.zeros(len(cal_df))
    for node_name in all_nodes:
        vals = cal_df[node_name].values
        tau = process_configs[node_name]['base_target']
        s = process_configs[node_name]['scale']
        scores += (vals - tau) ** 2 / max(s, 1e-8)

    best_idx = int(np.argmin(scores))
    ds.cal_reference_row = {col: float(cal_df[col].iloc[best_idx]) for col in cal_df.columns}


# ---------------------------------------------------------------------------
# Pre-built instances (Section 11)
# ---------------------------------------------------------------------------

ds_scm_st_minimal = build_st_scm(STConfig(
    n=2, m=1, me=0, rho=0.0,
))

ds_scm_st_simple = build_st_scm(STConfig(
    n=4, m=1, me=1, env_mode="A", rho=0.2,
))

ds_scm_st_medium = build_st_scm(STConfig(
    n=8, m=3, width_profile="exp_back", width_beta=0.5,
    me=2, env_mode="C", env_overlap=0.0, rho=0.4, p=1,
))

ds_scm_st_deep = build_st_scm(STConfig(
    n=12, m=4, width_profile="gaussian", width_mu=2.5, width_s=1.0,
    me=3, env_mode="D", env_overlap=0.2, rho=0.5, p=1,
))

ds_scm_st_multi = build_st_scm(STConfig(
    n=10, m=2, me=2, env_mode="B", rho=0.3, p=2, output_overlap=0.3,
))
