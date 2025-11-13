# Noise Classification Analysis for Multi-Scenario Controller Training

## Purpose
This document classifies noise variables in each SCM dataset into two categories:
1. **Structural Noise**: Environmental/operating conditions that create scenario diversity
2. **Process Noise**: Measurement/actuator imperfections that should be zeroed in target trajectories

## Methodology

For each SCM dataset, I examined:
1. The SCM graph structure (NodeSpec definitions)
2. The physical model equations and comments
3. The `noise_model.singles` dictionary entries
4. The input/output labels used in controller optimization

**Key Principle**:
- Structural noise variables should be **ACTIVE** in target trajectories (to create diverse training scenarios) and **SAME** between target and baseline (for fair comparison)
- Process noise variables should be **ZERO** in target trajectories (ideal behavior) and **ACTIVE** in baseline (realistic behavior)

---

## ds_scm_laser (Laser Drilling Process)

**Physical Model**: Light-Current-Temperature (L-I-T) model with heteroscedastic noise

**Process Inputs** (from process config):
- `PowerTarget`: Laser current setpoint [0.1 - 1.0 normalized]
- `AmbientTemp`: Environmental temperature [15 - 35°C]

### Classification:

**Structural Noise Variables**:
- `AmbientTemp` (eps_AmbientTemp)
  - **Rationale**: External environmental condition that varies between manufacturing sessions
  - **Physical meaning**: Ambient temperature affects laser efficiency and threshold current
  - **Sampling**: `rng.uniform(low=15.0, high=35.0, size=n)` (20°C range)
  - **Why structural**: Not controllable; represents different facility conditions, seasonal variation, etc.

**Process Noise Variables**:
- `Zln` (eps_Zln)
  - **Type**: Lognormal multiplicative gain fluctuations
  - **Sampling**: `rng.standard_normal(n)`
  - **Physical meaning**: Quantum efficiency variations

- `NoiseShot` (eps_NoiseShot)
  - **Type**: Photon shot noise (quantum noise)
  - **Sampling**: `rng.standard_normal(n)` scaled by `sqrt(κ*Pclean)`
  - **Physical meaning**: Fundamental quantum measurement uncertainty

- `NoiseMeas` (eps_NoiseMeas)
  - **Type**: Additive measurement electronics noise
  - **Sampling**: `rng.standard_normal(n)` scaled by `σa`
  - **Physical meaning**: Photodetector and electronics noise

- `NoiseDrift` (eps_NoiseDrift)
  - **Type**: Thermal drift AR(1) process
  - **Sampling**: `rng.standard_normal(n)` scaled by `σd/sqrt(1-ρ²)`
  - **Physical meaning**: Slow thermal variations in the laser cavity

**Notes**:
- `PowerTarget` is a control input (decision variable), not noise - it will be sampled/set during trajectory generation
- All physical constants (ETA0, ALPHA_T, etc.) are deterministic
- Intermediate nodes (TempDelta, I_th, Eff, Pclean, SigmaM) are deterministic computations

---

## ds_scm_plasma (Plasma Cleaning Process)

**Physical Model**: Exponential removal rate with plasma instability and micro-arcing events

**Process Inputs** (from process config):
- `RF_Power`: RF power setpoint [100 - 400 W]
- `Duration`: Process time [10 - 60 s]

### Classification:

**Structural Noise Variables**:
- **None identified in current model**
  - Note: The model includes a simplified heteroscedastic noise term `σm = σm0 + cP*(P/Pmax) + cp*|p - p0|`
  - The pressure term `p` is mentioned in comments but not implemented
  - Both RF_Power and Duration are controllable process parameters, not environmental conditions

**Process Noise Variables**:
- `Zln` (eps_Zln)
  - **Type**: Lognormal multiplicative plasma instability
  - **Sampling**: `rng.standard_normal(n)`
  - **Physical meaning**: Plasma density fluctuations

- `NoiseAdd` (eps_NoiseAdd)
  - **Type**: Additive measurement noise
  - **Sampling**: `rng.standard_normal(n)` scaled by `σa`
  - **Physical meaning**: Sensor measurement uncertainty

- `Jump` (eps_Jump)
  - **Type**: Poisson-driven micro-arcing jump events
  - **Sampling**: Custom - `Σ(k=1..K) Ak` where `K ~ Poisson(λJ)`, `Ak ~ Exp(θJ)`
  - **Physical meaning**: Rare but intense micro-arc discharges

**Notes**:
- This process has no environmental variation in the current model
- Scenario diversity will propagate from upstream processes (laser's ambient temperature variation affects ActualPower, which influences plasma inputs)

---

## ds_scm_galvanic (Galvanic Copper Deposition)

**Physical Model**: Faraday's law with spatial GP variation and electrical ripple

**Process Inputs** (from process config):
- `CurrentDensity`: Plating current density [1 - 5 A/dm²]
- `Duration`: Plating time [600 - 3600 s]

### Classification:

**Structural Noise Variables**:
- **None identified in current model**
  - Both CurrentDensity and Duration are controllable process parameters
  - No external environmental factors modeled

**Process Noise Variables**:
- `SpatialVar` (eps_SpatialVar)
  - **Type**: Gaussian spatial variation (simplified from Gaussian Process)
  - **Sampling**: `rng.standard_normal(n)` scaled by `σg`
  - **Physical meaning**: Non-uniform thickness across the board surface

- `TimeRand` (eps_TimeRand)
  - **Type**: Random time sample for electrical ripple
  - **Sampling**: `rng.uniform(low=0.0, high=1.0, size=n)`
  - **Physical meaning**: Random phase sampling for power supply ripple

- `PhaseRand` (eps_PhaseRand)
  - **Type**: Random phase for electrical ripple
  - **Sampling**: `rng.uniform(low=0.0, high=2*π, size=n)`
  - **Physical meaning**: Random initial phase of AC ripple

- `NoiseMeas` (eps_NoiseMeas)
  - **Type**: Additive measurement noise
  - **Sampling**: `rng.standard_normal(n)` scaled by `σa`
  - **Physical meaning**: Thickness measurement sensor uncertainty

**Notes**:
- The `Ripple` node is computed from TimeRand and PhaseRand, not a separate noise source
- `SpatialVar` appears twice in the singles dict (lines 507 and 513) - likely a typo, using the second definition

---

## ds_scm_microetch (Micro-Etching Process)

**Physical Model**: Arrhenius kinetics with concentration dependence and heavy-tailed surface roughening

**Process Inputs** (from process config):
- `Temperature`: Process/bath temperature [293 - 323 K] (20 - 50°C)
- `Concentration`: Etchant concentration [0.5 - 3.0 mol/L]
- `Duration`: Etching time [30 - 180 s]

### Classification:

**Structural Noise Variables**:
- `Temperature` (eps_Temperature)
  - **Rationale**: The wide range (30°C) suggests environmental/facility variation rather than tight process control
  - **Physical meaning**: Affects reaction rate via Arrhenius exponential `exp(-Ea/(R*T))`
  - **Sampling**: `rng.uniform(low=293.0, high=323.0, size=n)` (20-50°C)
  - **Why structural**: Different facilities, seasonal variations, or production batches may operate at different temperatures
  - **Impact**: 30°C temperature difference significantly affects etching rate due to exponential Arrhenius dependence

**Process Noise Variables**:
- `Zln` (eps_Zln)
  - **Type**: Lognormal multiplicative chemical variability
  - **Sampling**: `rng.standard_normal(n)` scaled by `σm = 0.06`
  - **Physical meaning**: Concentration fluctuations, mixing inhomogeneities

- `NoiseStudentT` (eps_NoiseStudentT)
  - **Type**: Heavy-tailed Student-t additive noise
  - **Sampling**: `rng.standard_t(df=5, size=n)` scaled by `st = 0.3 μm`
  - **Physical meaning**: Surface roughening with occasional large defects

**Notes**:
- `Concentration` and `Duration` are controllable process parameters
- Temperature classification is debatable - could be either structural (environmental) or controllable (bath setpoint)
- Chose structural based on the wide range and Arrhenius sensitivity

---

## Summary Table

| Dataset | Structural Noise Variables | Process Noise Variables |
|---------|---------------------------|-------------------------|
| **ds_scm_laser** | `AmbientTemp` | `Zln`, `NoiseShot`, `NoiseMeas`, `NoiseDrift` |
| **ds_scm_plasma** | *(none)* | `Zln`, `NoiseAdd`, `Jump` |
| **ds_scm_galvanic** | *(none)* | `SpatialVar`, `TimeRand`, `PhaseRand`, `NoiseMeas` |
| **ds_scm_microetch** | `Temperature` | `Zln`, `NoiseStudentT` |

---

## Implications for Multi-Scenario Training

### Scenario Diversity Sources:

1. **Direct structural variation**:
   - Laser: 50 different AmbientTemp values (15-35°C) → 50 distinct scenarios
   - Microetch: 50 different Temperature values (293-323K) → 50 distinct scenarios

2. **Propagated variation**:
   - Laser's AmbientTemp variation → affects ActualPower → influences downstream processes
   - Microetch's Temperature variation → affects RemovalDepth → influences final product quality

3. **Processes without structural noise** (plasma, galvanic):
   - Scenario diversity comes from upstream process variations
   - Example: Different laser ActualPower values (from different AmbientTemp scenarios) create different operating conditions for plasma

### Expected Training Behavior:

With multi-scenario training, the controller will learn to:
- **Laser**: Adapt PowerTarget based on AmbientTemp to achieve consistent ActualPower
- **Plasma**: Adapt RF_Power/Duration based on varying input conditions from laser
- **Galvanic**: Adapt CurrentDensity/Duration based on varying input conditions
- **Microetch**: Adapt Concentration/Duration based on Temperature variations and upstream variations

This ensures the controller generalizes across:
- Different environmental conditions (ambient temperature, process temperature)
- Cascading effects from upstream process variations
- Different operating points in the 4-process manufacturing chain

---

## Recommendations

1. **Consider adding structural noise to plasma and galvanic**:
   - Plasma: Could add ambient pressure or humidity variation
   - Galvanic: Could add electrolyte temperature or ambient humidity

2. **Temperature classification review**:
   - For microetch, confirm whether Temperature represents:
     - Bath temperature (tightly controlled) → treat as control input, not structural
     - Environmental/facility temperature → structural noise (current classification)

3. **Control inputs vs noise**:
   - Input variables like `PowerTarget`, `RF_Power`, etc. are NOT classified as noise
   - These will be handled separately as control/decision variables
   - The classification above focuses only on NOISE SOURCES in the `singles` dictionary

4. **Validation**:
   - After implementation, verify that target trajectories show:
     - High diversity in structural variables (many different AmbientTemp/Temperature values)
     - Near-zero variance in process noise variables (Zln, NoiseShot, etc. ≈ 0)

---

## Next Steps

**AWAITING USER CONFIRMATION** before proceeding to Phase 2 (implementation).

Please review this classification and confirm:
1. Agreement with structural vs process noise classification
2. Any changes needed (especially for `Temperature` in microetch)
3. Whether to add structural noise variables to plasma and galvanic processes
4. Approval to proceed with code implementation
