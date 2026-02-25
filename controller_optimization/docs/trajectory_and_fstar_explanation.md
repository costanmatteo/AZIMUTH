# Target Trajectory and F* Calculation — How They Drive Controller Training

## Overview

The controller optimization pipeline relies on three key concepts:

1. **Target trajectory (a\*)** — the ideal process behavior under no equipment noise
2. **F\* (target reliability)** — a scalar quality score computed from the target trajectory
3. **A training loop** that minimizes the gap between actual reliability F and F\*

---

## 1. Target Trajectory (a\*)

**Source**: `src/utils/target_generation.py` — `generate_target_trajectory()`

The target trajectory answers: *"What would the process chain produce under perfect conditions?"*

### Generation Procedure

For each process in the chain (laser → plasma → galvanic → microetch):

1. Load the process's Structural Causal Model (SCM) dataset
2. Modify the noise model:
   - **Structural noise** (e.g., `AmbientTemp`) → **kept active** to create diverse operating scenarios
   - **Process noise** (e.g., `Zln`, `NoiseMeas`, `NoiseDrift`) → **set to ~1e-6** (effectively zero)
3. Sample 50 scenarios with varying structural conditions
4. Record the inputs, outputs, and structural conditions for each scenario

### Result

```python
target_trajectory = {
    'laser': {
        'inputs': (50, input_dim),              # 50 scenarios of process inputs
        'outputs': (50, output_dim),            # deterministic ideal outputs
        'structural_conditions': {
            'AmbientTemp': [15.2, 16.1, ..., 34.8]
        }
    },
    'plasma': { ... },
    'galvanic': { ... },
    'microetch': { ... }
}
```

The structural conditions are saved so the baseline trajectory can be generated under identical environmental factors.

---

## 2. Baseline Trajectory (a')

**Source**: `src/utils/target_generation.py` — `generate_baseline_trajectory()`

The baseline answers: *"What happens without any controller intervention, under real-world noise?"*

- **Same inputs** as target (exactly aligned)
- **Same structural conditions** (temperature, etc.)
- **Process noise is active** — realistic equipment jitter, measurement drift, etc.

This serves as the "no intervention" benchmark.

---

## 3. F\* — Target Reliability Score

**Source**: `src/models/surrogate.py` — `compute_reliability()` and `_compute_F_star_from_scenario_0()`

F\* is the reliability of the target trajectory — a single scalar representing the best quality the controller can aim for.

### Quality Scoring Per Process

Each process has a Gaussian quality function:

```
Q_i(output) = exp( -(output - τ_i)² / s_i )
```

| Process   | Target (τ) | Scale (s) | Weight | Output Measured   |
|-----------|-----------|-----------|--------|-------------------|
| laser     | 0.8       | 0.1       | 1.0    | ActualPower       |
| plasma    | 3.0       | 2.0       | 1.0    | RemovalRate       |
| galvanic  | 10.0      | 4.0       | 1.5    | Thickness (μm)    |
| microetch | 20.0      | 4.0       | 1.0    | Depth             |

**Adaptive targets**: Downstream targets adjust based on upstream outputs. For example:

- Plasma target: `τ_plasma = 3.0 + 0.2 × (laser_power - 0.8)`
- Galvanic target: `τ_galvanic = 10.0 + 0.5 × (plasma_rate - 5.0) + 0.4 × (laser_power - 0.5)`
- Microetch target: `τ_microetch = 20.0 + 1.5 × (laser_power - 0.5) + 0.3 × (plasma_rate - 5.0) - 0.15 × (galvanic_thick - 10.0)`

### Aggregation to F\*

```
F* = Σ(weight_i × Q_i) / Σ(weight_i)
```

Galvanic carries the highest weight (1.5), reflecting its importance in the manufacturing chain.

F\* is computed from scenario 0 of the target trajectory and remains fixed throughout training.

---

## 4. Controller Training

**Source**: `src/training/controller_trainer.py`

### 4.1 Loss Function

```
L = reliability_weight × 100 × (F - F*)² + λ_BC × BC_loss
```

| Term | Purpose |
|------|---------|
| `(F - F*)²` | Push actual reliability toward target. Scaled by 100 to prevent vanishing gradients. |
| `BC_loss` | Behavior cloning: MSE between controller's chosen inputs and the target trajectory's inputs (normalized). Anchors the controller near known-good operating points. |

### 4.2 Forward Pass (How F Is Computed)

For each scenario during training:

1. **Initial inputs** are taken from the target trajectory for the given scenario
2. Pass through the **uncertainty predictor** for the first process → get (output_mean, output_variance)
3. **Sample stochastically** using the reparameterization trick: `output = mean + ε × √variance` (differentiable)
4. Feed sampled output into the **policy generator** (the trainable controller), which chooses controllable inputs for the next process
5. Merge with non-controllable inputs from the target trajectory
6. Repeat through the full chain (laser → plasma → galvanic → microetch)
7. Compute F from the complete trajectory using the same quality function as F\*

### 4.3 Curriculum Learning

Training uses a phased schedule:

| Phase | Epochs | Reliability Weight | λ_BC | Goal |
|-------|--------|--------------------|------|------|
| Warmup | First ~10% | 0 | 1.0 | Learn to mimic target inputs |
| Curriculum | Remaining ~90% | 0 → 1.0 (S-curve) | 1.0 → 0.05 (exp. decay) | Shift from imitation to optimization |

The S-curve for reliability weight uses: `reliability_weight = 1 - exp(-5 × progress)`

The exponential decay for BC uses: `λ_BC = λ_start × exp(3 × ln(λ_end/λ_start) × progress)`

### 4.4 Multi-Scenario Training Loop

Each epoch:

1. Shuffles all 50 scenarios randomly
2. For each scenario, runs a forward pass and computes the loss
3. Accumulates gradients (scaled by 1/50 to average across scenarios)
4. Takes a **single optimizer step** after processing all scenarios

This gradient accumulation ensures consistent learning across all operating conditions.

---

## 5. Evaluation: Comparing the Three Reliability Metrics

| Metric | Symbol | Computed From | Meaning |
|--------|--------|---------------|---------|
| Target reliability | F\* | Target trajectory (no noise) | Best achievable quality |
| Baseline reliability | F' | Baseline trajectory (with noise, no controller) | Uncontrolled performance |
| Actual reliability | F | Forward pass with trained controller | Controlled performance |

### Success Criteria

- **Improvement**: `(F - F') / F' > 0` — controller is better than doing nothing
- **Optimality gap**: `(F* - F) / F* < 5%` — controller is close to the ideal
- **Robustness**: `std(F across scenarios) < 0.01` — consistent across operating conditions

---

## 6. Key Files Reference

| Component | File | Key Functions/Lines |
|-----------|------|---------------------|
| Target trajectory generation | `src/utils/target_generation.py` | `generate_target_trajectory()` (L53–160) |
| Baseline trajectory generation | `src/utils/target_generation.py` | `generate_baseline_trajectory()` (L163–250) |
| F\* and quality computation | `src/models/surrogate.py` | `compute_reliability()` (L91–248), `_compute_F_star_from_scenario_0` (L250–270) |
| Process chain forward pass | `src/utils/process_chain.py` | Forward pass (L485–661) |
| Loss function | `src/training/controller_trainer.py` | `compute_loss()` (L425–493) |
| Training loop | `src/training/controller_trainer.py` | `train_epoch()` (L495–677) |
| Curriculum schedule | `src/training/controller_trainer.py` | Curriculum logic (L230–297) |
| Configuration | `configs/controller_config.py` | All hyperparameters (L97–231) |
