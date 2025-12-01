# Theoretical Loss Analysis for Reliability-based Controller Optimization

## Overview

This module implements a theoretical framework for computing the **minimum achievable loss (L_min)** when training a controller using stochastic sampling from an UncertaintyPredictor. The key insight is that when the controller's output is sampled stochastically (using the reparameterization trick), there exists an **irreducible minimum loss** that cannot be reduced regardless of how well the controller is trained.

---

## 1. Theoretical Framework

### 1.1 The Reliability Loss Function

The controller is trained to minimize:

```
L = scale × (F - F*)²
```

Where:
- **F*** = reliability of the target trajectory (deterministic, computed from expert demonstrations)
- **F** = reliability of the controller's trajectory (stochastic, due to sampling)
- **scale** = loss scaling factor (default: 100.0)

### 1.2 The Quality Function

For each process i, the quality is computed as:

```
Q_i(o) = exp(-(o - τ_i)² / s_i)
```

Where:
- **o** = process output (e.g., ActualPower for laser, RemovalRate for plasma)
- **τ_i** = target value for optimal quality (process-specific)
- **s_i** = scale parameter controlling sensitivity (smaller = more sensitive)

### 1.3 Stochastic Sampling

The controller uses an UncertaintyPredictor that outputs:
- **μ** = predicted mean output
- **σ²** = predicted variance

The actual output is sampled using the **reparameterization trick**:

```
o = μ + σ × ε,    where ε ~ N(0, 1)
```

This introduces stochasticity into the quality computation.

---

## 2. Mathematical Derivation

### 2.1 Key Parameters

For each process i:
- **δ_i = μ_target,i - τ_i** : distance between target output and process optimum
- **σ²_i** : predicted variance from UncertaintyPredictor
- **s_i** : scale parameter of quality function
- **F*_i = exp(-δ_i²/s_i)** : target quality for process i

### 2.2 Expected Value of Quality

When sampling o ~ N(μ_target, σ²), the expected quality is:

```
E[Q] = F* × (1/√(1 + 2σ²/s)) × exp(2δ²σ² / (s(s + 2σ²)))
```

**Breakdown of factors:**
1. **F*** : baseline quality at target
2. **1/√(1 + 2σ²/s)** : reduction due to variance (always < 1 when σ² > 0)
3. **exp(...)** : correction for off-center target (> 1 when δ ≠ 0)

### 2.3 Expected Value of Quality Squared

From Corollary 16 of the theoretical document:

Since Q² = exp(-2(δ+σε)²/s) has effective scale s/2, applying Lemma 9
with a=4δσ/s and b=2σ²/s gives:

```
E[Q²] = F*² × (1/√(1 + 4σ²/s)) × exp(8δ²σ² / (s(s + 4σ²)))
```

**Note**: The exponent numerator is **8** (not 4). This is because:
- a² = (4δσ/s)² = 16δ²σ²/s²
- 1 + 2b = 1 + 4σ²/s = (s + 4σ²)/s
- a²/(2(1+2b)) = 16δ²σ²/s² × s/(2(s + 4σ²)) = 8δ²σ² / (s(s + 4σ²))

### 2.4 Minimum Achievable Loss

The loss can be decomposed as:

```
E[L] = E[(F - F*)²] = Var[F] + Bias²
```

Where:
- **Var[F] = E[F²] - E[F]²** : irreducible variance due to stochastic sampling
- **Bias² = (E[F] - F*)²** : systematic bias (reducible through better control)

The **theoretical minimum** is:

```
L_min = Var[F] + Bias²
```

When σ² > 0, we have **L_min > 0** because:
1. Var[F] > 0 (stochastic sampling always adds variance)
2. Bias² ≥ 0 (may be zero if E[F] = F*)

---

## 3. Implementation Architecture

### 3.1 Module Structure

```
controller_optimization/src/analysis/
├── __init__.py                      # Module exports
├── theoretical_loss_analysis.py     # Core formulas and tracker
├── theoretical_visualization.py     # Plot generation
├── theoretical_tables.py            # Table generation
└── THEORETICAL_LOSS_ANALYSIS.md     # This documentation
```

### 3.2 Key Classes and Functions

#### `TheoreticalLossComponents` (dataclass)
Stores all components of the theoretical analysis:
```python
@dataclass
class TheoreticalLossComponents:
    L_min: float      # Minimum achievable loss
    E_F: float        # E[F]
    E_F2: float       # E[F²]
    Var_F: float      # Var[F] = E[F²] - E[F]²
    Bias2: float      # (E[F] - F*)²
    F_star: float     # Target reliability
    sigma2: float     # Predicted variance
    delta: float      # Distance from optimum
    s: float          # Scale parameter
```

#### `compute_theoretical_E_F(F_star, delta, sigma2, s)`
Computes E[F] using the theoretical formula.

#### `compute_theoretical_E_F2(F_star, delta, sigma2, s)`
Computes E[F²] using the theoretical formula.

#### `compute_theoretical_L_min(F_star, delta, sigma2, s, loss_scale)`
Computes all theoretical components and returns `TheoreticalLossComponents`.

#### `compute_multi_process_L_min(process_params, process_weights, loss_scale)`
Combines per-process L_min values using weighted averaging.

#### `TheoreticalLossTracker`
Tracks theoretical vs observed loss throughout training:
- Records per-epoch: observed_loss, L_min, gap, efficiency
- Computes empirical statistics from validation sampling
- Detects violations (observed < theoretical)

---

## 4. Step-by-Step Flow During Training

### Step 1: Initialize Tracker (train_controller.py, Step 5.5/9)

```python
theoretical_tracker = TheoreticalLossTracker(
    loss_scale=CONTROLLER_CONFIG['training']['reliability_loss_scale']
)

# Load process configs (τ, s) from surrogate
for proc_name, proc_config in ProTSurrogate.PROCESS_CONFIGS.items():
    theoretical_tracker.process_configs[proc_name] = {
        'tau': proc_config['target'],
        's': proc_config['scale']
    }
```

### Step 2: Run Training (Steps 6-8)

Training proceeds normally. The reliability loss is recorded in `history['reliability_loss']`.

### Step 3: Validation Sampling (Step 8.6/9)

After training, run multiple forward passes to collect empirical statistics:

```python
F_samples_all = []
sigma2_per_process = {proc_name: [] for proc_name in active_processes}

with torch.no_grad():
    for _ in range(n_validation_samples):
        for scenario_idx in range(n_scenarios):
            trajectory = process_chain.forward(batch_size=1, scenario_idx=scenario_idx)
            F = surrogate.compute_reliability(trajectory).item()
            F_samples_all.append(F)

            # Collect σ² per process
            for proc_name, data in trajectory.items():
                sigma2_per_process[proc_name].append(data['outputs_var'].mean().item())
```

### Step 4: Compute Per-Process Parameters

For each active process:

```python
for proc_name in active_processes:
    cfg = ProTSurrogate.PROCESS_CONFIGS[proc_name]
    tau = cfg['target']      # Process optimum (e.g., 0.8 for laser)
    s = cfg['scale']         # Quality scale (e.g., 0.1 for laser)

    # Get target output from target_trajectory
    mu_target = target_trajectory[proc_name]['outputs'].mean()

    # CORRECT delta calculation
    delta = mu_target - tau  # e.g., 0.75 - 0.8 = -0.05

    # Per-process F*
    F_star_i = np.exp(-delta**2 / s)  # e.g., exp(-0.0025/0.1) = 0.9753

    # Mean σ² for this process
    sigma2_i = np.mean(sigma2_per_process[proc_name])

    process_params[proc_name] = {
        'F_star': F_star_i,
        'delta': delta,
        'sigma2': sigma2_i,
        's': s
    }
```

### Step 5: Compute Combined L_min

```python
combined_components, per_process_components = compute_multi_process_L_min(
    process_params=process_params,
    process_weights={'laser': 1.0, 'plasma': 1.0},  # from PROCESS_CONFIGS
    loss_scale=100.0
)

# Result: combined_components.L_min > 0
```

### Step 6: Populate Tracker History

For each training epoch:

```python
for epoch, (rel_loss, F_val) in enumerate(zip(reliability_loss_history, F_values_history)):
    theoretical_tracker.update(
        epoch=epoch,
        observed_loss_value=rel_loss,
        F_star=F_star_mean,
        F_samples=np.array([F_val]),
        sigma2_mean=combined_sigma2,
        delta=combined_delta,
        s=combined_s
    )
```

### Step 7: Generate Outputs

```python
# Get all data
theoretical_data = theoretical_tracker.to_dict()

# Generate plots
generate_all_theoretical_plots(theoretical_data, checkpoint_dir)

# Generate text report
generate_full_report(theoretical_data, process_params)

# Save JSON
save_report_json(theoretical_data, checkpoint_dir / 'theoretical_analysis_data.json')
```

---

## 5. Output Interpretation

### 5.1 Console Output During Training

```
[8.6/9] Running theoretical loss analysis...
  Computing per-process theoretical parameters...
  Running validation sampling for empirical statistics...
  Validation samples: 250
  Empirical E[F]: 0.892345
  Empirical Var[F]: 0.00012345

  Per-process theoretical parameters:
    laser:
      τ (target) = 0.8000, μ_target = 0.7500
      δ = μ_target - τ = -0.0500
      σ² = 0.000006, s = 0.1000
      F*_i = exp(-δ²/s) = 0.9753
    plasma:
      τ (target) = 3.0000, μ_target = 2.8000
      δ = μ_target - τ = -0.2000
      σ² = 0.000010, s = 2.0000
      F*_i = exp(-δ²/s) = 0.9802

  Combined theoretical L_min: 0.001234
    Var[F] component: 0.000456
    Bias² component:  0.000778

  THEORETICAL ANALYSIS SUMMARY:
    Final Loss:        0.012188
    Theoretical L_min: 0.001234
    Gap (reducible):   0.010954
    Efficiency:        10.1%
    Violations:        0/100
```

### 5.2 Key Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **L_min** | Minimum achievable loss | > 0 (with σ² > 0) |
| **Gap** | observed - L_min | As small as possible |
| **Efficiency** | L_min / observed | 100% = optimal |
| **Violations** | Times observed < L_min | 0 (theory validated) |

### 5.3 Interpreting Results

1. **High Efficiency (>90%)**: Controller is near-optimal; further training won't help much
2. **Low Efficiency (<50%)**: Room for improvement; continue training
3. **Violations > 0**: Either numerical issues or theory assumptions violated
4. **Large Var[F]**: High stochasticity; consider reducing σ² predictions
5. **Large Bias²**: Systematic offset; controller needs better alignment

---

## 6. Generated Files

After training completes, the following files are generated in `checkpoint_dir/`:

| File | Description |
|------|-------------|
| `theoretical_analysis_data.json` | Complete tracker data in JSON format |
| `theoretical_analysis_report.txt` | Human-readable text report |
| `theoretical_loss_vs_observed.png` | Loss over epochs with L_min line |
| `theoretical_efficiency.png` | Efficiency percentage over epochs |
| `theoretical_loss_decomposition.png` | Var[F] vs Bias² bar chart |
| `theoretical_scatter.png` | Observed vs theoretical scatter plot |
| `theoretical_summary.png` | 2x2 grid of all plots |

---

## 7. Mathematical Validation

### 7.1 Sanity Checks

The implementation includes validation:

1. **σ² = 0 ⟹ L_min = 0**: Deterministic case, no irreducible loss
2. **σ² > 0 ⟹ L_min > 0**: Stochastic case, always has minimum
3. **Observed ≥ L_min**: If violated, indicates theory issues
4. **E[F] ≤ F*** (typically): Variance usually reduces expected quality

### 7.2 Unit Test

```python
# Test with known values
F_star = 0.85
delta = 0.2
sigma2 = 0.05
s = 1.0

components = compute_theoretical_L_min(F_star, delta, sigma2, s)

assert components.L_min > 0, "L_min should be positive with σ² > 0"
assert components.Var_F >= 0, "Variance cannot be negative"
assert components.E_F <= F_star * 1.1, "E[F] should be close to F*"
```

---

## 8. Process Configuration Reference

From `ProTSurrogate.PROCESS_CONFIGS`:

| Process | τ (target) | s (scale) | weight |
|---------|-----------|-----------|--------|
| laser | 0.8 | 0.1 | 1.0 |
| plasma | 3.0 | 2.0 | 1.0 |
| galvanic | 10.0 | 4.0 | 1.5 |
| microetch | 20.0 | 4.0 | 1.0 |

**Note**: Smaller `s` means higher sensitivity to deviations from τ.

---

## 9. Common Issues and Solutions

### Issue: L_min = 0

**Cause**: Incorrect delta calculation (was estimating from F* instead of target_trajectory)

**Solution**: Compute δ = μ_target - τ directly from target_trajectory outputs

### Issue: Very small L_min despite σ² > 0

**Check**:
1. Is σ²/s very small? (e.g., σ² = 0.000001, s = 0.1 → ratio = 0.00001)
2. Is δ ≈ 0? (target already at process optimum)

### Issue: Observed < L_min (violations)

**Possible causes**:
1. Numerical precision issues
2. Non-Gaussian sampling distribution
3. Finite sample bias

---

## 10. References

- Reparameterization trick: Kingma & Welling (2014), "Auto-Encoding Variational Bayes"
- Bias-variance decomposition: Geman et al. (1992), "Neural Networks and the Bias/Variance Dilemma"
