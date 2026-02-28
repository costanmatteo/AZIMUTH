# Empirical Loss Analysis for Reliability-based Controller Optimization

## Overview

This module collects **empirical statistics** (E[F], Var[F], Bias²) from stochastic forward passes through the process chain. These statistics characterize the behavior of the reliability function F under sampling uncertainty.

---

## 1. Framework

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

## 2. Empirical Statistics

### 2.1 Collected Statistics

From N stochastic forward passes, we compute:

- **E[F] = mean(F_samples)**: expected reliability
- **Var[F] = var(F_samples)**: variance of reliability due to sampling
- **Bias² = (E[F] - F*)²**: systematic offset from target

### 2.2 Interpretation

The loss can be decomposed as:

```
E[L] = E[(F - F*)²] = Var[F] + Bias²
```

Where:
- **Var[F]**: variance due to stochastic sampling
- **Bias²**: systematic bias (E[F] ≠ F*)

When σ² > 0:
1. Var[F] > 0 (stochastic sampling always adds variance)
2. Bias² ≥ 0 (may be zero if E[F] = F*)

---

## 3. Implementation Architecture

### 3.1 Module Structure

```
controller_optimization/src/analysis/
├── __init__.py                      # Module exports
├── theoretical_loss_analysis.py     # Core statistics and tracker
├── theoretical_visualization.py     # Plot generation
├── theoretical_tables.py            # Table generation
├── bellman_lmin.py                  # Bellman backward-induction L_min
└── THEORETICAL_LOSS_ANALYSIS.md     # This documentation
```

### 3.2 Key Classes and Functions

#### `EmpiricalStats` (dataclass)
Stores empirical statistics:
```python
@dataclass
class EmpiricalStats:
    E_F: float        # E[F]
    E_F2: float       # E[F²]
    Var_F: float      # Var[F] = E[F²] - E[F]²
    Bias2: float      # (E[F] - F*)²
    F_star: float     # Target reliability
```

#### `compute_empirical_stats(F_samples, F_star, loss_scale)`
Computes empirical statistics from forward-pass samples.

#### `TheoreticalLossTracker`
Tracks observed loss and empirical statistics throughout training:
- Records per-epoch: observed_loss, empirical E[F], Var[F], Bias²
- Provides summary at end of training

---

## 4. Step-by-Step Flow During Training

### Step 1: Initialize Tracker (train_controller.py, Step 5.5/9)

```python
theoretical_tracker = TheoreticalLossTracker(
    loss_scale=CONTROLLER_CONFIG['training']['reliability_loss_scale']
)
```

### Step 2: Run Training (Steps 6-8)

Training proceeds normally. The reliability loss is recorded in `history['reliability_loss']`.

### Step 3: Validation Sampling (Step 8.6/9)

After training, run multiple forward passes to collect empirical statistics:

```python
F_samples_all = []
with torch.no_grad():
    for _ in range(n_repeats):
        for scenario_idx in range(n_scenarios):
            trajectory = process_chain.forward(batch_size=samples_per_scenario, scenario_idx=scenario_idx)
            F = surrogate.compute_reliability(trajectory)
            F_samples_all.extend(F.tolist())

empirical_stats = compute_empirical_stats(
    F_samples=np.array(F_samples_all),
    F_star=surrogate.F_star,
    loss_scale=loss_scale
)
```

### Step 4: Generate Outputs

```python
theoretical_data = theoretical_tracker.to_dict()

# Generate plots
generate_all_theoretical_plots(theoretical_data, checkpoint_dir)

# Generate text report
generate_full_report(theoretical_data, process_params)
```

---

## 5. Output Interpretation

### 5.1 Console Output During Training

```
[8.6/9] Running theoretical loss analysis...
  Running validation sampling (empirical statistics)...
  Total F samples: 125000

  Empirical statistics (from 125000 samples):
    E[F]:             0.892345
    Var[F] (scaled):  0.012345
    Bias² (scaled):   0.007800
    F*:               0.950000

  ANALYSIS SUMMARY:
    Final Loss:   0.012188
    Best Loss:    0.011500
    E[F]:         0.892345
    Var[F]:       0.012345
    Bias²:        0.007800
```

### 5.2 Key Metrics

| Metric | Description |
|--------|-------------|
| **E[F]** | Expected reliability from stochastic sampling |
| **Var[F]** | Variance of reliability (from sampling noise) |
| **Bias²** | Systematic offset (E[F] - F*)² |

### 5.3 Interpreting Results

1. **Large Var[F]**: High stochasticity; controller predictions have high uncertainty
2. **Large Bias²**: Systematic offset from target; controller consistently over/under-shoots
3. **Var[F] ≈ 0**: Deterministic behavior (low prediction uncertainty)
4. **Bias² ≈ 0**: E[F] matches F* well (no systematic bias)

---

## 6. Generated Files

After training completes, the following files are generated in `checkpoint_dir/`:

| File | Description |
|------|-------------|
| `theoretical_analysis_data.json` | Complete tracker data in JSON format |
| `theoretical_analysis_report.txt` | Human-readable text report |
| `observed_loss.png` | Observed loss over epochs (+ Bellman L_min if available) |
| `loss_decomposition.png` | Var[F] vs Bias² bar chart |
| `theoretical_analysis_summary.png` | Summary figure |

---

## 7. Bellman L_min (Separate Analysis)

The Bellman backward-induction L_min provides an independent theoretical lower bound
on achievable loss via dynamic programming. See `bellman_lmin.py` for details.

When computed, the Bellman L_min line appears on the observed loss plot.

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

## 9. References

- Reparameterization trick: Kingma & Welling (2014), "Auto-Encoding Variational Bayes"
- Bias-variance decomposition: Geman et al. (1992), "Neural Networks and the Bias/Variance Dilemma"
