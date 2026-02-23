# L_min Calculation — Detailed Conceptual Explanation

## What is L_min?

L_min is the **theoretical minimum achievable loss** when training the controller with
stochastic sampling from the UncertaintyPredictor. It represents the irreducible lower
bound on the reliability loss — even with a perfect controller, the stochasticity of
sampling prevents the loss from reaching zero.

---

## Step-by-step calculation

### Step 1: Per-process parameters

For each process i in the manufacturing chain:

```
τ_i     = target value for optimal quality (from PROCESS_CONFIGS)
s_i     = scale parameter of quality function (from PROCESS_CONFIGS)
w_i     = weight of process i (from PROCESS_CONFIGS)
μ_target,i = output from expert/target trajectory
δ_i     = μ_target,i - τ_i       (distance from process optimum)
Q_i*    = exp(-δ_i² / s_i)       (target quality for process i)
σ²_i    = predicted variance from UncertaintyPredictor
```

### Step 2: Expected value of quality (Theorem 10)

When output o is sampled from N(μ_target, σ²), the expected quality is:

```
E[Q_i] = Q_i* × (1/√(1 + 2σ²/s)) × exp(2δ²σ² / (s(s + 2σ²)))
```

The three factors represent:
1. **Q_i*** — baseline quality at target
2. **1/√(1 + 2σ²/s)** — reduction due to variance (always < 1 for σ² > 0)
3. **exp(...)** — correction for off-center target (≥ 1 when δ ≠ 0)

### Step 3: Expected value of quality squared (Corollary 16)

Since Q² = exp(-2(δ+σε)²/s) has effective scale s/2:

```
E[Q_i²] = Q_i*² × (1/√(1 + 4σ²/s)) × exp(8δ²σ² / (s(s + 4σ²)))
```

**Critical**: The exponent numerator is **8** (not 4). Derivation:
- a = 4δσ/s, b = 2σ²/s
- a²/(2(1+2b)) = 16δ²σ²/s² × s/(2(s + 4σ²)) = 8δ²σ² / (s(s + 4σ²))

### Step 4: Per-process variance

```
Var[Q_i] = E[Q_i²] - E[Q_i]²
```

### Step 5: Combined reliability (multi-process)

The overall reliability is a weighted average:

```
F  = Σ(w_i × Q_i) / W        where W = Σ w_i
F* = Σ(w_i × Q_i*) / W
```

Expected values:

```
E[F] = Σ(w_i × E[Q_i]) / W
```

### Step 6: Combined variance (Theorem 27/31)

**Independent processes** (ρ_ij = 0):

```
Var[F] = Σ(w_i² × Var[Q_i]) / W²
```

**Correlated processes** (Theorem 45, Corollary 46):

```
Var[F] = (1/W²) × Σ_i Σ_j w_i w_j Cov(Q_i, Q_j)
```

### Step 7: Bias-variance decomposition

```
Bias² = (E[F] - F*)²
```

### Step 8: Final L_min

```
L_min = (Var[F] + Bias²) × loss_scale
```

where loss_scale defaults to 100.0.

---

## Why L_min > 0 when σ² > 0

Two irreducible components:

1. **Var[F] > 0**: Stochastic sampling always adds variance to the quality score.
   Even with perfect control, each sample of ε ~ N(0,1) produces a different output,
   hence a different Q, hence a different F.

2. **Bias² ≥ 0**: The expected value E[F] is generally not equal to F* because
   the quality function Q is nonlinear (Gaussian). Taking the expectation of a
   nonlinear function of a random variable introduces a systematic shift
   (Jensen's inequality effect).

---

## Usage during training

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Gap | Loss_observed - L_min | Reducible portion of loss |
| Efficiency | L_min / Loss_observed | How close to theoretical optimum |
| Violations | Count(Loss < L_min) | Theory validation check |

- **Efficiency > 90%**: controller is near-optimal
- **Efficiency < 50%**: significant room for improvement
- **Violations > 0**: indicates theoretical framework issues

---

## Key code references

- Core formulas: `theoretical_loss_analysis.py` (functions at lines 66, 102, 150, 367)
- Process configs: `surrogate.py:38` (PROCESS_CONFIGS)
- Quality function: `surrogate.py:162` (laser), `:179` (plasma), `:200` (galvanic), `:225` (microetch)
- Adaptive targets: `surrogate.py:152-226`
- Diagnostic tool: `diagnostic_L_min.py`

## Referenced theorems

- **Theorem 10**: E[Q] formula
- **Corollary 16**: E[Q²] formula (exponent is 8δ²σ², NOT 4δ²σ²)
- **Theorem 27**: Variance propagation for weighted combinations
- **Theorem 31**: Independence assumption (ρ_ij = 0)
- **Theorem 45**: Cross-moment E[Q_i Q_j] for correlated processes
- **Corollary 43**: Quadratic form reduction for cross-moments
- **Corollary 46**: Cov(Q_i, Q_j) = E[Q_i Q_j] - E[Q_i]E[Q_j]
