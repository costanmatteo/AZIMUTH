


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
# LASER DRILLING SCM MODEL — Section 2 of PDF
# -----------------------------------------------------------------------------
# Reference: appunti_processi.pdf, Section 2 (Laser Drilling Model)
#
# This dataset models the actual optical power of a laser drilling system
# following a physically-grounded Light-Current-Temperature (L-I-T) model
# with process-specific stochastic noise.
#
# -------------------------
# DETERMINISTIC COMPONENT (Eq. 1):
#   Pclean(I, T) = η0 * (1 - αT*(T - T0)) * [I - I0*e^(kT*(T-T0))]
#
# Where:
#   I  = PowerTarget (normalized current setpoint)
#   T  = AmbientTemp (temperature in °C)
#   η0 = quantum efficiency at reference temperature
#   αT = efficiency temperature coefficient
#   I0 = threshold current at T0
#   kT = threshold temperature coefficient
#   T0 = reference temperature
#
# -------------------------
# NOISE MODEL (Eq. 2-7):
#   Pactual(t) = Pclean(t) * Zln(t) + εshot(t) + εmeas(t) + d(t)
#
# Components:
#   1. Zln(t) ~ LogNormal(-σ²m/2, σ²m)     [multiplicative gain fluctuations]
#   2. εshot(t) ~ N(0, κ*Pclean(t))        [photon shot noise]
#   3. εmeas(t) ~ N(0, σ²a)                [measurement electronics noise]
#   4. d(t) = ρ*d(t-1) + ηt, ηt ~ N(0,σ²d) [thermal drift AR(1)]
#
# Heteroscedastic noise scale (Eq. 7):
#   σm = σm0 + cT*|T - T0| + cI*(I/Imax)
#
# -------------------------
# PARAMETERS (suggested ranges from PDF):
#   η0   = 1.0          # quantum efficiency
#   αT   = 0.005        # efficiency temp coefficient [1/°C]
#   I0   = 0.10         # threshold current
#   kT   = 0.03         # threshold temp coefficient [1/°C]
#   T0   = 25.0         # reference temperature [°C]
#   σm0  = 0.02         # base multiplicative noise std
#   cT   = 0.001        # temp-dependent noise slope
#   cI   = 0.03         # current-dependent noise slope
#   κ    = 0.001        # shot noise coefficient
#   σa   = 0.01         # measurement noise std
#   σd   = 0.005        # drift innovation std
#   ρ    = 0.95         # AR(1) drift coefficient
#   Imax = 1.0          # normalized max current
# =============================================================================

ds_scm_laser = SCMDataset(
    name="laser_drilling_physical",
    description="Laser drilling optical power with L-I-T model and physical noise (lognormal, shot, measurement, AR drift)",
    tags=["laser", "drilling", "pcb", "physical"],
    specs=[
        # ==================== INPUTS ====================
        NodeSpec("PowerTarget", [], "eps_PowerTarget"),
        NodeSpec("AmbientTemp", [], "eps_AmbientTemp"),

        # ==================== PHYSICAL CONSTANTS ====================
        NodeSpec("ETA0",     [], "eps_ETA0"),
        NodeSpec("ALPHA_T",  [], "eps_ALPHA_T"),
        NodeSpec("I0",       [], "eps_I0"),
        NodeSpec("K_T",      [], "eps_K_T"),
        NodeSpec("T0",       [], "eps_T0"),
        NodeSpec("SIGMA_M0", [], "eps_SIGMA_M0"),
        NodeSpec("C_T",      [], "eps_C_T"),
        NodeSpec("C_I",      [], "eps_C_I"),
        NodeSpec("KAPPA",    [], "eps_KAPPA"),
        NodeSpec("SIGMA_A",  [], "eps_SIGMA_A"),
        NodeSpec("SIGMA_D",  [], "eps_SIGMA_D"),
        NodeSpec("RHO",      [], "eps_RHO"),
        NodeSpec("IMAX",     [], "eps_IMAX"),

        # ==================== DETERMINISTIC EQUATION (Eq. 1) ====================
        # Temperature delta
        NodeSpec("TempDelta", ["AmbientTemp", "T0"],
                 "AmbientTemp - T0 + 0*eps_TempDelta"),

        # Threshold current I_th = I0 * exp(kT * ΔT)
        NodeSpec("I_th", ["I0", "K_T", "TempDelta"],
                 "I0 * exp(K_T * TempDelta) + 0*eps_I_th"),

        # Efficiency η = η0 * (1 - αT * ΔT)
        NodeSpec("Eff", ["ETA0", "ALPHA_T", "TempDelta"],
                 "ETA0 * (1 - ALPHA_T * TempDelta) + 0*eps_Eff"),

        # Clean power Pclean = η * (I - I_th)
        NodeSpec("Pclean", ["Eff", "PowerTarget", "I_th"],
                 "Eff * (PowerTarget - I_th) + 0*eps_Pclean"),

        # ==================== HETEROSCEDASTIC NOISE SCALE (Eq. 7) ====================
        NodeSpec("SigmaM", ["SIGMA_M0", "C_T", "TempDelta", "C_I", "PowerTarget", "IMAX"],
                 "SIGMA_M0 + C_T*abs(TempDelta) + C_I*(PowerTarget/IMAX) + 0*eps_SigmaM"),

        # ==================== NOISE COMPONENTS ====================
        # 1. Lognormal multiplicative noise Zln ~ LogNormal(-σ²m/2, σ²m) [Eq. 3]
        #    Transform: Zln = exp(-σ²m/2 + σm * ε) where ε ~ N(0,1)
        NodeSpec("Zln", ["SigmaM"], "exp(-SigmaM**2 / 2.0 + SigmaM * eps_Zln)"),

        # 2. Shot noise εshot ~ N(0, κ*Pclean) [Eq. 4]
        #    Standard deviation = sqrt(κ*Pclean)
        NodeSpec("NoiseShot", ["Pclean", "KAPPA"],
                 "sqrt(KAPPA * abs(Pclean)) * eps_NoiseShot + 0*Pclean"),

        # 3. Measurement noise εmeas ~ N(0, σ²a) [Eq. 5]
        NodeSpec("NoiseMeas", ["SIGMA_A"], "SIGMA_A * eps_NoiseMeas"),

        # 4. Thermal drift d(t) ~ AR(1) [Eq. 6]
        #    For i.i.d. samples, we use stationary distribution: N(0, σ²d/(1-ρ²))
        NodeSpec("NoiseDrift", ["SIGMA_D", "RHO"],
                 "SIGMA_D / sqrt(1 - RHO**2) * eps_NoiseDrift"),

        # ==================== OUTPUT (Eq. 2) ====================
        # Pactual = Pclean * Zln + εshot + εmeas + d
        NodeSpec("ActualPower", ["Pclean", "Zln", "NoiseShot", "NoiseMeas", "NoiseDrift"],
                 "Pclean * Zln + NoiseShot + NoiseMeas + NoiseDrift + 0*eps_ActualPower"),
    ],
    params={},
    singles={
        # ==================== RANDOM INPUTS ====================
        "PowerTarget": lambda rng, n: rng.uniform(low=0.10, high=1.0, size=n),
        "AmbientTemp": lambda rng, n: rng.uniform(low=15.0, high=35.0, size=n),

        # ==================== CONSTANTS (deterministic) ====================
        "ETA0":     lambda rng, n: np.full(n, 1.0),
        "ALPHA_T":  lambda rng, n: np.full(n, 0.005),
        "I0":       lambda rng, n: np.full(n, 0.10),
        "K_T":      lambda rng, n: np.full(n, 0.03),
        "T0":       lambda rng, n: np.full(n, 25.0),
        "SIGMA_M0": lambda rng, n: np.full(n, 0.02),
        "C_T":      lambda rng, n: np.full(n, 0.001),
        "C_I":      lambda rng, n: np.full(n, 0.03),
        "KAPPA":    lambda rng, n: np.full(n, 0.001),
        "SIGMA_A":  lambda rng, n: np.full(n, 0.01),
        "SIGMA_D":  lambda rng, n: np.full(n, 0.005),
        "RHO":      lambda rng, n: np.full(n, 0.95),
        "IMAX":     lambda rng, n: np.full(n, 1.0),

        # ==================== INTERMEDIATE NODES (deterministic) ====================
        "TempDelta": lambda rng, n: np.zeros(n),
        "I_th":      lambda rng, n: np.zeros(n),
        "Eff":       lambda rng, n: np.zeros(n),
        "Pclean":    lambda rng, n: np.zeros(n),
        "SigmaM":    lambda rng, n: np.zeros(n),

        # ==================== NOISE SOURCES ====================
        # Lognormal: standard normal ε ~ N(0,1), transformed to LogN in node equation
        "Zln": lambda rng, n: rng.standard_normal(n),

        # Shot noise: standard normal (will be scaled by sqrt(κ*Pclean) in equation)
        "NoiseShot": lambda rng, n: rng.standard_normal(n),

        # Measurement noise: standard normal (will be scaled by σa in equation)
        "NoiseMeas": lambda rng, n: rng.standard_normal(n),

        # Drift noise: standard normal (will be scaled by σd/sqrt(1-ρ²) in equation)
        "NoiseDrift": lambda rng, n: rng.standard_normal(n),

        # ==================== OUTPUT ====================
        "ActualPower": lambda rng, n: np.zeros(n),
    },
    groups=None,
    input_labels=["PowerTarget", "AmbientTemp"],
    target_labels=["ActualPower"],
)

# =============================================================================
# NOISE CLASSIFICATION FOR MULTI-SCENARIO TRAINING
# -----------------------------------------------------------------------------
# Structural noise: Environmental conditions that create scenario diversity
# Process noise: Measurement/actuator imperfections (zeroed in target trajectory)
# =============================================================================
ds_scm_laser.structural_noise_vars = ['AmbientTemp']
ds_scm_laser.process_noise_vars = ['Zln', 'NoiseShot', 'NoiseMeas', 'NoiseDrift']


# =============================================================================
# PLASMA CLEANING SCM MODEL — Section 3 of PDF
# -----------------------------------------------------------------------------
# Reference: appunti_processi.pdf, Section 3 (Plasma Cleaning Model)
#
# This dataset models surface residue removal rate in plasma cleaning
# with multiplicative variability and rare micro-arcing jump events.
#
# -------------------------
# DETERMINISTIC COMPONENT (SIMPLIFIED):
#   Original (Eq. 8): Rclean = k0 * exp(-λp * P^β) * τ
#   Simplified to: Rclean = k0 * P * τ / 1000
#
# This linearization removes two non-linearities (exponential decay + power function)
# to simplify controller optimization while maintaining input-output relationships.
#
# Where:
#   P    = RF_Power (RF power in W, range: 100-400)
#   τ    = Duration (process time in s, range: 10-60)
#   k0   = base removal rate coefficient (0.5)
#
# -------------------------
# NOISE MODEL (Eq. 9-10):
#   Ractual = Rclean * Zln + εa + J
#
# Components:
#   1. Zln ~ LogNormal(-σ²m/2, σ²m)              [plasma instability]
#   2. εa ~ N(0, σ²a)                            [additive measurement noise]
#   3. J = Σ(k=1..K) Ak, K ~ Poisson(λJ), Ak ~ Exp(θJ)  [micro-arcing jumps]
#
# Heteroscedastic noise scale (Eq. 10):
#   σm = σm0 + cP*(P/Pmax) + cp*|p - p0|
#   (where p is pressure, simplified here)
#
# -------------------------
# PARAMETERS:
#   k0   = 0.5          # base removal rate coefficient
#   σm0  = 0.03         # base multiplicative noise
#   cP   = 0.02         # power-dependent noise
#   σa   = 0.01         # additive noise
#   λJ   = 0.1          # Poisson rate for arcing events
#   θJ   = 0.05         # Exponential scale for arc amplitude
#   Pmax = 500          # max RF power [W] (for noise scaling)
#
# REMOVED (simplification):
#   λp   = 0.02         # power decay coefficient (was in exponential)
#   β    = 0.8          # power exponent (was in exponential)
# =============================================================================

ds_scm_plasma = SCMDataset(
    name="plasma_cleaning_physical",
    description="Plasma cleaning residue removal with physical noise (lognormal + Poisson micro-arcing jumps)",
    tags=["plasma", "cleaning", "pcb", "physical"],
    specs=[
        # ==================== INPUTS ====================
        NodeSpec("RF_Power", [], "eps_RF_Power"),
        NodeSpec("Duration", [], "eps_Duration"),

        # ==================== PHYSICAL CONSTANTS ====================
        NodeSpec("K0",       [], "eps_K0"),
        NodeSpec("SIGMA_M0", [], "eps_SIGMA_M0"),
        NodeSpec("C_P",      [], "eps_C_P"),
        NodeSpec("SIGMA_A",  [], "eps_SIGMA_A"),
        NodeSpec("LAMBDA_J", [], "eps_LAMBDA_J"),
        NodeSpec("THETA_J",  [], "eps_THETA_J"),
        NodeSpec("PMAX",     [], "eps_PMAX"),

        # ==================== DETERMINISTIC EQUATION (SIMPLIFIED) ====================
        # Rclean = k0 * P * τ / 1000  (simplified from exponential to linear)
        NodeSpec("Rclean", ["K0", "RF_Power", "Duration"],
                 "K0 * RF_Power * Duration / 1000.0 + 0*eps_Rclean"),

        # ==================== HETEROSCEDASTIC NOISE SCALE (Eq. 10) ====================
        NodeSpec("SigmaM", ["SIGMA_M0", "C_P", "RF_Power", "PMAX"],
                 "SIGMA_M0 + C_P * (RF_Power / PMAX) + 0*eps_SigmaM"),

        # ==================== NOISE COMPONENTS ====================
        # 1. Lognormal multiplicative noise Zln ~ LogNormal(-σ²m/2, σ²m) [Eq. 9]
        NodeSpec("Zln", ["SigmaM"], "exp(-SigmaM**2 / 2.0 + SigmaM * eps_Zln)"),

        # 2. Additive measurement noise εa ~ N(0, σ²a)
        NodeSpec("NoiseAdd", ["SIGMA_A"], "SIGMA_A * eps_NoiseAdd"),

        # 3. Poisson-driven micro-arcing jumps [Eq. 9]
        #    J = Σ(k=1..K) Ak, K ~ Poisson(λJ), Ak ~ Exp(θJ)
        NodeSpec("Jump", [], "eps_Jump"),

        # ==================== OUTPUT (Eq. 9) ====================
        # Ractual = Rclean * Zln + εa + J
        NodeSpec("RemovalRate", ["Rclean", "Zln", "NoiseAdd", "Jump"],
                 "Rclean * Zln + NoiseAdd + Jump + 0*eps_RemovalRate"),
    ],
    params={},
    singles={
        # ==================== RANDOM INPUTS ====================
        "RF_Power": lambda rng, n: rng.uniform(low=100.0, high=400.0, size=n),
        "Duration": lambda rng, n: rng.uniform(low=10.0, high=60.0, size=n),

        # ==================== CONSTANTS ====================
        "K0":       lambda rng, n: np.full(n, 0.5),
        "SIGMA_M0": lambda rng, n: np.full(n, 0.03),
        "C_P":      lambda rng, n: np.full(n, 0.02),
        "SIGMA_A":  lambda rng, n: np.full(n, 0.01),
        "LAMBDA_J": lambda rng, n: np.full(n, 0.1),
        "THETA_J":  lambda rng, n: np.full(n, 0.05),
        "PMAX":     lambda rng, n: np.full(n, 500.0),

        # ==================== INTERMEDIATE NODES ====================
        "Rclean":   lambda rng, n: np.zeros(n),
        "SigmaM":   lambda rng, n: np.zeros(n),

        # ==================== NOISE SOURCES ====================
        # Lognormal: standard normal (transformed in equation)
        "Zln": lambda rng, n: rng.standard_normal(n),

        # Additive noise: standard normal (scaled in equation)
        "NoiseAdd": lambda rng, n: rng.standard_normal(n),

        # Poisson jumps: custom sampler
        "Jump": lambda rng, n: np.array([
            np.sum(rng.exponential(0.05, rng.poisson(0.1))) if rng.poisson(0.1) > 0 else 0.0
            for _ in range(n)
        ]),

        # ==================== OUTPUT ====================
        "RemovalRate": lambda rng, n: np.zeros(n),
    },
    groups=None,
    input_labels=["RF_Power", "Duration"],
    target_labels=["RemovalRate"],
)

# Noise classification: No structural noise (scenario diversity from upstream processes)
ds_scm_plasma.structural_noise_vars = []
ds_scm_plasma.process_noise_vars = ['Zln', 'NoiseAdd', 'Jump']


# =============================================================================
# GALVANIC COPPER DEPOSITION SCM MODEL — Section 4 of PDF
# -----------------------------------------------------------------------------
# Reference: appunti_processi.pdf, Section 4 (Electrolytic Copper Deposition)
#
# This dataset models deposited copper thickness in electrolytic plating
# with spatial variation (GP field) and electrical ripple noise.
#
# -------------------------
# DETERMINISTIC COMPONENT (Eq. 11):
#   tCu,ideal = (ηdep * j * τ * MCu) / (n * F * ρCu)
#
# Where:
#   j    = CurrentDensity (A/dm²)
#   τ    = Duration (time in seconds)
#   ηdep = deposition efficiency (0-1)
#   MCu  = molar mass of Cu (63.546 g/mol)
#   n    = electrons per Cu atom (2)
#   F    = Faraday constant (96485 C/mol)
#   ρCu  = copper density (8.96 g/cm³)
#
# -------------------------
# NOISE MODEL (Eq. 12):
#   tCu(x, y, t) = tCu,ideal * (1 + g(x,y)) + ar*sin(2πfr*t + φ) + εa
#
# Components:
#   1. g ~ GP(0, σ²g * exp(-|r|²/(2ℓ²)))  [spatial thickness variation]
#   2. ar*sin(2πfr*t + φ)                  [electrical ripple]
#   3. εa ~ N(0, σ²a)                      [measurement noise]
#
# For i.i.d. samples, we simplify:
#   - g ~ N(0, σ²g) [independent spatial samples]
#   - t, φ ~ Uniform [random time and phase for ripple]
#
# -------------------------
# PARAMETERS:
#   ηdep = 0.95         # deposition efficiency
#   MCu  = 63.546       # Cu molar mass [g/mol]
#   n    = 2            # electrons per Cu atom
#   F    = 96485        # Faraday constant [C/mol]
#   ρCu  = 8.96         # Cu density [g/cm³]
#   σg   = 0.02         # spatial variation std
#   ar   = 0.01         # ripple amplitude [μm]
#   fr   = 100          # ripple frequency [Hz]
#   σa   = 0.005        # measurement noise [μm]
# =============================================================================

ds_scm_galvanic = SCMDataset(
    name="galvanic_copper_deposition_physical",
    description="Galvanic Cu deposition thickness with spatial GP-like variation and electrical ripple",
    tags=["galvanic", "copper", "deposition", "pcb", "physical"],
    specs=[
        # ==================== INPUTS ====================
        NodeSpec("CurrentDensity", [], "eps_CurrentDensity"),
        NodeSpec("Duration",       [], "eps_Duration"),

        # ==================== PHYSICAL CONSTANTS ====================
        NodeSpec("ETA_DEP", [], "eps_ETA_DEP"),
        NodeSpec("M_CU",    [], "eps_M_CU"),
        NodeSpec("N_ELEC",  [], "eps_N_ELEC"),
        NodeSpec("FARADAY", [], "eps_FARADAY"),
        NodeSpec("RHO_CU",  [], "eps_RHO_CU"),
        NodeSpec("SIGMA_G", [], "eps_SIGMA_G"),
        NodeSpec("A_R",     [], "eps_A_R"),
        NodeSpec("F_R",     [], "eps_F_R"),
        NodeSpec("SIGMA_A", [], "eps_SIGMA_A"),

        # ==================== DETERMINISTIC EQUATION (Eq. 11) ====================
        # tCu,ideal = (ηdep * j * τ * MCu) / (n * F * ρCu)
        # Units: (A/dm²) * s * (g/mol) / (mol/C) / (g/cm³) = (A*s*g/mol) / (mol*g/cm³*C)
        # Conversion factor: 1 A = 1 C/s, 1 dm² = 100 cm²
        # Result in μm: multiply by 10000 (cm to μm) and divide by 100 (dm² to cm²)
        # Factor: 100 for dm² → cm²
        NodeSpec("tCu_ideal", ["ETA_DEP", "CurrentDensity", "Duration", "M_CU", "N_ELEC", "FARADAY", "RHO_CU"],
                 "100.0 * ETA_DEP * CurrentDensity * Duration * M_CU / (N_ELEC * FARADAY * RHO_CU) + 0*eps_tCu_ideal"),

        # ==================== NOISE COMPONENTS ====================
        # 1. Spatial variation g ~ N(0, σ²g) [simplified from GP]
        NodeSpec("SpatialVar", ["SIGMA_G"], "SIGMA_G * eps_SpatialVar"),

        # 2. Electrical ripple: ar * sin(2π*fr*t + φ)
        #    We sample random time t and phase φ for each sample
        NodeSpec("TimeRand",  [], "eps_TimeRand"),   # Random time for ripple
        NodeSpec("PhaseRand", [], "eps_PhaseRand"),  # Random phase for ripple
        NodeSpec("Ripple", ["A_R", "F_R", "TimeRand", "PhaseRand"],
                 "A_R * sin(2 * 3.14159265359 * F_R * TimeRand + PhaseRand) + 0*eps_Ripple"),

        # 3. Measurement noise εa ~ N(0, σ²a)
        NodeSpec("NoiseMeas", ["SIGMA_A"], "SIGMA_A * eps_NoiseMeas"),

        # ==================== OUTPUT (Eq. 12) ====================
        # tCu = tCu,ideal * (1 + g) + ripple + εa
        NodeSpec("Thickness", ["tCu_ideal", "SpatialVar", "Ripple", "NoiseMeas"],
                 "tCu_ideal * (1 + SpatialVar) + Ripple + NoiseMeas + 0*eps_Thickness"),
    ],
    params={},
    singles={
        # ==================== RANDOM INPUTS ====================
        "CurrentDensity": lambda rng, n: rng.uniform(low=1.0, high=5.0, size=n),
        "Duration":       lambda rng, n: rng.uniform(low=600.0, high=3600.0, size=n),

        # ==================== CONSTANTS ====================
        "ETA_DEP": lambda rng, n: np.full(n, 0.95),
        "M_CU":    lambda rng, n: np.full(n, 63.546),
        "N_ELEC":  lambda rng, n: np.full(n, 2.0),
        "FARADAY": lambda rng, n: np.full(n, 96485.0),
        "RHO_CU":  lambda rng, n: np.full(n, 8.96),
        "SIGMA_G": lambda rng, n: np.full(n, 0.02),
        "A_R":     lambda rng, n: np.full(n, 0.01),
        "F_R":     lambda rng, n: np.full(n, 100.0),
        "SIGMA_A": lambda rng, n: np.full(n, 0.005),

        # ==================== INTERMEDIATE NODES ====================
        "tCu_ideal":  lambda rng, n: np.zeros(n),
        "SpatialVar": lambda rng, n: np.zeros(n),
        "Ripple":     lambda rng, n: np.zeros(n),
        "NoiseMeas":  lambda rng, n: np.zeros(n),

        # ==================== NOISE SOURCES ====================
        # Spatial variation: standard normal (scaled in equation)
        "SpatialVar": lambda rng, n: rng.standard_normal(n),

        # Random time for ripple (e.g., 0-1 second)
        "TimeRand": lambda rng, n: rng.uniform(low=0.0, high=1.0, size=n),

        # Random phase for ripple (0-2π)
        "PhaseRand": lambda rng, n: rng.uniform(low=0.0, high=2*np.pi, size=n),

        # Measurement noise: standard normal (scaled in equation)
        "NoiseMeas": lambda rng, n: rng.standard_normal(n),

        # ==================== OUTPUT ====================
        "Thickness": lambda rng, n: np.zeros(n),
    },
    groups=None,
    input_labels=["CurrentDensity", "Duration"],
    target_labels=["Thickness"],
)

# Noise classification: No structural noise (scenario diversity from upstream processes)
ds_scm_galvanic.structural_noise_vars = []
ds_scm_galvanic.process_noise_vars = ['SpatialVar', 'TimeRand', 'PhaseRand', 'NoiseMeas']


# =============================================================================
# MICRO-ETCHING SCM MODEL — Section 5 of PDF
# -----------------------------------------------------------------------------
# Reference: appunti_processi.pdf, Section 5 (Micro-Etching Model)
#
# This dataset models copper surface removal/roughening in micro-etching
# with lognormal variability and heavy-tailed Student-t perturbations.
#
# -------------------------
# DETERMINISTIC COMPONENT (Eq. 13):
#   Rremoved = ketch * e^(-Ea/(R*T)) * C^α * τ
#
# Where:
#   T     = Temperature (Kelvin)
#   C     = Concentration (mol/L)
#   τ     = Duration (seconds)
#   ketch = pre-exponential factor
#   Ea    = activation energy (J/mol)
#   R     = gas constant (8.314 J/(mol·K))
#   α     = concentration exponent
#
# -------------------------
# NOISE MODEL (Eq. 14):
#   Ractual = Rremoved * Zln + εt
#
# Components:
#   1. Zln ~ LogNormal(-σ²m/2, σ²m)       [multiplicative chemical variability]
#   2. εt ~ Student-t(ν, 0, st)           [heavy-tailed surface roughening]
#
# -------------------------
# PARAMETERS:
#   ketch = 1.0         # pre-exponential factor [μm/s]
#   Ea    = 50000       # activation energy [J/mol]
#   R     = 8.314       # gas constant [J/(mol·K)]
#   α     = 0.5         # concentration exponent
#   σm    = 0.04        # multiplicative noise std
#   ν     = 5           # Student-t degrees of freedom
#   st    = 0.02        # Student-t scale [μm]
# =============================================================================

ds_scm_microetch = SCMDataset(
    name="microetch_physical",
    description="Micro-etching Cu removal with Arrhenius kinetics and heavy-tailed Student-t noise",
    tags=["microetch", "etching", "pcb", "physical"],
    specs=[
        # ==================== INPUTS ====================
        NodeSpec("Temperature",   [], "eps_Temperature"),
        NodeSpec("Concentration", [], "eps_Concentration"),
        NodeSpec("Duration",      [], "eps_Duration"),

        # ==================== PHYSICAL CONSTANTS ====================
        NodeSpec("K_ETCH",   [], "eps_K_ETCH"),
        NodeSpec("E_A",      [], "eps_E_A"),
        NodeSpec("R_GAS",    [], "eps_R_GAS"),
        NodeSpec("ALPHA",    [], "eps_ALPHA"),
        NodeSpec("SIGMA_M",  [], "eps_SIGMA_M"),
        NodeSpec("NU",       [], "eps_NU"),
        NodeSpec("S_T",      [], "eps_S_T"),

        # ==================== DETERMINISTIC EQUATION ====================
        NodeSpec("Rremoved", ["K_ETCH", "E_A", "R_GAS", "Temperature", "Concentration", "ALPHA", "Duration"],
                 "K_ETCH * exp(-E_A / (R_GAS * Temperature)) * Concentration**ALPHA * Duration + 0*eps_Rremoved"),

        # ==================== NOISE COMPONENTS ====================
        NodeSpec("Zln", ["SIGMA_M"], "exp(-SIGMA_M**2 / 2.0 + SIGMA_M * eps_Zln)"),
        NodeSpec("NoiseStudentT", ["S_T"], "S_T * eps_NoiseStudentT"),

        # ==================== OUTPUT ====================
        NodeSpec("RemovalDepth", ["Rremoved", "Zln", "NoiseStudentT"],
                 "Rremoved * Zln + NoiseStudentT + 0*eps_RemovalDepth"),
    ],
    params={},
    singles={
        # Inputs
        "Temperature":   lambda rng, n: rng.uniform(low=293.0, high=323.0, size=n),
        "Concentration": lambda rng, n: rng.uniform(low=0.5, high=3.0, size=n),
        "Duration":      lambda rng, n: rng.uniform(low=30.0, high=180.0, size=n),

        # RECOMMENDED: Realistic activation energy
        "K_ETCH":  lambda rng, n: np.full(n, 8e5),      # Pre-exponential factor
        "E_A":     lambda rng, n: np.full(n, 42000.0),  # 42 kJ/mol - typical for acid etching
        "R_GAS":   lambda rng, n: np.full(n, 8.314),
        "ALPHA":   lambda rng, n: np.full(n, 0.5),      # Square-root concentration dependence

        # Noise parameters
        "SIGMA_M": lambda rng, n: np.full(n, 0.06),     # 6% multiplicative noise
        "NU":      lambda rng, n: np.full(n, 5.0),      # Student-t degrees of freedom
        "S_T":     lambda rng, n: np.full(n, 0.3),      # 0.3 μm additive noise

        # ==================== INTERMEDIATE & NOISE NODES ====================
        "Rremoved": lambda rng, n: np.zeros(n),
        "Zln": lambda rng, n: rng.standard_normal(n),
        "NoiseStudentT": lambda rng, n: rng.standard_t(df=5, size=n),
        "RemovalDepth": lambda rng, n: np.zeros(n),
    },
    groups=None,
    input_labels=["Temperature", "Concentration", "Duration"],
    target_labels=["RemovalDepth"],
)

# Noise classification: Temperature as structural (environmental/facility variation)
ds_scm_microetch.structural_noise_vars = ['Temperature']
ds_scm_microetch.process_noise_vars = ['Zln', 'NoiseStudentT']



# =============================================================================
# EXAMPLE USAGE (commented out)
# =============================================================================
# Uncomment and modify as needed to generate datasets

# ds_scm_1_to_1_ct.generate_ds(mode="flat", n=5_000, save_dir=join(ROOT_DIR, "data/one_to_one"))
# ds_scm_laser.generate_ds(mode="flat", n=5_000, save_dir=join(ROOT_DIR, "data/laser"))
# ds_scm_plasma.generate_ds(mode="flat", n=5_000, save_dir=join(ROOT_DIR, "data/plasma"))
# ds_scm_galvanic.generate_ds(mode="flat", n=5_000, save_dir=join(ROOT_DIR, "data/galvanic"))
# ds_scm_microetch.generate_ds(mode="flat", n=5_000, save_dir=join(ROOT_DIR, "data/microetch"))
