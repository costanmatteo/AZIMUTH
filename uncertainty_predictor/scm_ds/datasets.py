


from os.path import abspath, dirname
import sys

ROOT_DIR = dirname(dirname(abspath(__file__)))
print("Root directory: ", ROOT_DIR)
sys.path.append(ROOT_DIR)

from scm_ds.scm import *



ds_scm_1_to_1_ct = SCMDataset(
    name = "one-to-one_with_crosstalk",
    description ="Every parent has one child and there is cross-talk between children",
    tags=None,
    specs = [
        NodeSpec("P1", [], "eps_P1"),                  # parent 1
        NodeSpec("P2", [], "eps_P2"),                  # parent 2
        NodeSpec("P3", [], "eps_P3"),                  # parent 3
        NodeSpec("P4", [], "eps_P4"),                  # parent 4
        NodeSpec("P5", [], "eps_P5"),                  # parent 5
        NodeSpec("C1", ["P1"], "P1 + eps_C1"),                  # child 1
        NodeSpec("C2", ["P2"], "P2 + eps_C2"),                  # child 2
        NodeSpec("C3", ["P3"], "P3 + eps_C3"),                  # child 3
        NodeSpec("C4", ["P4"], "P4 + eps_C4"),                  # child 4
        NodeSpec("C5", ["P5"], "P5 + eps_C5"),                  # child 5
        # output
        NodeSpec("Y", ["C1", "C2", "C3", "C4", "C5"],    "C1 + C2 + C3 + C4 + C5 + eps_Y"),     
        ],
    params = {
        "w1": 0.01,
        "w2": 0.01,
        "w3": 0.01,
        "w4": 0.01,
        "w5": 0.01,
        },
    singles = {
        "P1": lambda rng,n: rng.standard_normal(n),
        "P2": lambda rng,n: rng.standard_normal(n),
        "P3": lambda rng,n: rng.standard_normal(n),
        "P4": lambda rng,n: rng.standard_normal(n),
        "P5": lambda rng,n: rng.standard_normal(n),
        "C1": lambda rng,n: rng.standard_normal(n),
        "C2": lambda rng,n: rng.standard_normal(n),
        "C3": lambda rng,n: rng.standard_normal(n),
        "C4": lambda rng,n: rng.standard_normal(n),
        "C5": lambda rng,n: rng.standard_normal(n),
        "Y": lambda rng,n: rng.standard_normal(n),
        },
    groups=None,
    input_labels=[
        "P1", "P2", "P3", "P4", "P5",
        "C1", "C2", "C3", "C4", "C5"],
    target_labels = ["Y"]
    )
    

# TODO
# - one-to-one-CT
# - one-to-many-noCT
# - one-to-many-noCT


# Example usage (commented out to avoid running on import)
# ds_scm_1_to_1_ct.generate_ds(mode="flat", n=5_000, save_dir=join(ROOT_DIR, "data/example"))


















# =============================================================================
# LASER SCM MODEL — Equations Reference
# -----------------------------------------------------------------------------
# This dataset models the "ActualPower" of a laser system as a function of:
#   PowerTarget (I) and AmbientTemp (T)
# using a Light–Current–Temperature (L–I–T) relation with
# additive + multiplicative heteroscedastic noise.
#
# -------------------------
# Inputs:
#   I  = PowerTarget       # normalized current setpoint
#   T  = AmbientTemp       # temperature (°C)
#
# -------------------------
# Intermediates:
#   ΔT        = T - T0
#   I_th      = I0 * exp(k_T * ΔT)
#   η         = η0 * (1 - α_T * ΔT)
#   P_clean   = η * (I - I_th)
#
# -------------------------
# Noise amplitudes (heteroscedastic):
#   σ_m(T,I) = σ_m0 + c_T * |ΔT| + c_I * (I / I_max)
#   σ_a(I)   = σ_a0 * P_FS + d_I * P_FS * (I / I_max)
#
# Random terms:
#   ε_m ~ N(0,1)     # multiplicative noise
#   ε_a ~ N(0,1)     # additive noise
#
# -------------------------
# Output equation:
#   P_actual = P_clean * (1 + σ_m * ε_m) + σ_a * ε_a
#
# Optional clipping:
#   P_actual_clipped = max(0, P_actual)
#
# -------------------------
# Parameters (example defaults):
#   η0      = 1.0
#   α_T     = 0.005        # efficiency loss per °C
#   I0      = 0.1          # threshold current (normalized)
#   k_T     = 0.03         # threshold growth per °C
#   T0      = 25.0         # reference temperature (°C)
#   σ_m0    = 0.01         # base multiplicative noise
#   c_T     = 0.0005       # temp-dependent noise slope
#   c_I     = 0.01         # current-dependent noise slope
#   σ_a0    = 0.005        # base additive noise (fraction FS)
#   d_I     = 0.002        # current-dependent additive noise
#   I_max   = 1.0          # normalized max current
#   P_FS    = 1.0          # full-scale power (for normalization)
#
# Reference:
#   H. Takamizawa et al., "Temperature Dependence of Threshold Current
#   and Slope Efficiency of Semiconductor Lasers",
#   IEEE J. Quantum Electronics, vol. 43, no. 12, pp. 1161–1167, 2007.
# =============================================================================




from os.path import abspath, dirname, join
import sys
import numpy as np

ROOT_DIR = dirname(dirname(abspath(__file__)))
print("Root directory: ", ROOT_DIR)
sys.path.append(ROOT_DIR)

from scm_ds.scm import SCMDataset, NodeSpec  # avoid wildcard

# =========================
# DATASET: LASER ACTUAL POWER
# =========================

ds_scm_1_to_1_ct = SCMDataset(
    name="laser_actual_power",
    description="ActualPower from PowerTarget (I) and AmbientTemp (T) via L-I-T with additive & multiplicative noise",
    tags=None,
    specs=[
        # Inputs
        NodeSpec("PowerTarget", [], "0.0*eps_I_base + 1.0 * U_I"),
        NodeSpec("AmbientTemp", [], "0.0*eps_T_base + 25.0 + 3.0*U_T"),

        # Constants
        NodeSpec("ETA0",     [], "0.0*eps_c0  + 1.0"),
        NodeSpec("ALPHA_T",  [], "0.0*eps_c1  + 0.005"),
        NodeSpec("I0",       [], "0.0*eps_c2  + 0.10"),
        NodeSpec("K_T",      [], "0.0*eps_c3  + 0.03"),
        NodeSpec("T0",       [], "0.0*eps_c4  + 25.0"),
        NodeSpec("SIGMA_M0", [], "0.0*eps_c5  + 0.01"),
        NodeSpec("C_T",      [], "0.0*eps_c6  + 0.0005"),
        NodeSpec("C_I",      [], "0.0*eps_c7  + 0.01"),
        NodeSpec("SIGMA_A0", [], "0.0*eps_c8  + 0.005"),
        NodeSpec("D_I",      [], "0.0*eps_c9  + 0.002"),
        NodeSpec("IMAX",     [], "0.0*eps_c10 + 1.0"),
        NodeSpec("PFS",      [], "0.0*eps_c11 + 1.0"),

        # Intermediates
        NodeSpec("TempDelta", ["AmbientTemp", "T0"], "AmbientTemp - T0"),
        NodeSpec("I_th",      ["I0", "K_T", "TempDelta"], "I0 * exp(K_T * TempDelta)"),
        NodeSpec("Eff",       ["ETA0", "ALPHA_T", "TempDelta"], "ETA0 * (1 - ALPHA_T * TempDelta)"),
        NodeSpec("Pclean",    ["Eff", "PowerTarget", "I_th"], "Eff * (PowerTarget - I_th)"),

        # Heteroscedastic scales
        NodeSpec("SigmaM", ["SIGMA_M0", "C_T", "TempDelta", "C_I", "PowerTarget", "IMAX"],
                "SIGMA_M0 + C_T*Abs(TempDelta) + C_I*(PowerTarget/IMAX)"),
        NodeSpec("SigmaA", ["SIGMA_A0", "D_I", "PFS", "PowerTarget", "IMAX"],
                 "SIGMA_A0*PFS + D_I*PFS*(PowerTarget/IMAX)"),

        # Output
        NodeSpec("ActualPower", ["Pclean", "SigmaM", "SigmaA"],
                 "Pclean * (1 + SigmaM * eps_M) + SigmaA * eps_A"),
    ],
    params={},
    singles={
        # Inputs
        "U_I": lambda rng, n: rng.uniform(low=0.05, high=1.0, size=n),
        "U_T": lambda rng, n: rng.uniform(low=-6.0, high=+6.0, size=n),
        "eps_I_base": lambda rng, n: rng.standard_normal(n),
        "eps_T_base": lambda rng, n: rng.standard_normal(n),

        # Noises
        "eps_M": lambda rng, n: rng.standard_normal(n),
        "eps_A": lambda rng, n: rng.standard_normal(n),

        # Placeholders for constants
        "eps_c0":  lambda rng, n: rng.standard_normal(n),
        "eps_c1":  lambda rng, n: rng.standard_normal(n),
        "eps_c2":  lambda rng, n: rng.standard_normal(n),
        "eps_c3":  lambda rng, n: rng.standard_normal(n),
        "eps_c4":  lambda rng, n: rng.standard_normal(n),
        "eps_c5":  lambda rng, n: rng.standard_normal(n),
        "eps_c6":  lambda rng, n: rng.standard_normal(n),
        "eps_c7":  lambda rng, n: rng.standard_normal(n),
        "eps_c8":  lambda rng, n: rng.standard_normal(n),
        "eps_c9":  lambda rng, n: rng.standard_normal(n),
        "eps_c10": lambda rng, n: rng.standard_normal(n),
        "eps_c11": lambda rng, n: rng.standard_normal(n),
    },
    groups=None,
    input_labels=["PowerTarget", "AmbientTemp"],
    target_labels=["ActualPower"],
)

# Example:
# ds_scm_1_to_1_ct.generate_ds(mode="flat", n=50_000, save_dir=join(ROOT_DIR, "data/laser_example"))
