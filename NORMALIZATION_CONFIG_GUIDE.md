# Normalization Configuration Guide

## Overview

The policy generator output normalization can be configured through the `controller_config.py` file. These settings control how the policy's normalized outputs [0, 1] are mapped to actual process input ranges.

## Configuration Parameters

### Location

In `controller_optimization/configs/controller_config.py`:

```python
'policy_generator': {
    ...
    # Output normalization settings
    'range_estimation_method': 'uncertainty_predictor',  # or 'target_trajectory'
    'range_scale_factor': 3.0,  # For StandardScaler: k in [mean - k*scale, mean + k*scale]
},
```

### Parameters

#### `range_estimation_method`

**Type:** `str`
**Default:** `'uncertainty_predictor'`
**Options:** `'uncertainty_predictor'` | `'target_trajectory'`

Determines the source for computing input ranges used in denormalization.

**'uncertainty_predictor'** (Recommended):
- Uses statistics from the uncertainty predictor's training data
- For StandardScaler: computes range as `[mean - k×scale, mean + k×scale]`
- For MinMaxScaler: uses stored `data_min_` and `data_max_`
- **Advantages:**
  - ✅ Works correctly with any number of scenarios (even 1)
  - ✅ Uses full operational range of each process (5000+ training samples)
  - ✅ Physically meaningful (actual process capabilities)
  - ✅ Robust and independent of target trajectory sampling

**'target_trajectory'** (Legacy):
- Computes min/max from target trajectory scenario samples
- **WARNING:** Degenerates with 1 scenario (min=max → no learning)
- Only use with multiple scenarios (20+)
- Kept for backward compatibility

**Example:**
```python
# Recommended for all use cases
'range_estimation_method': 'uncertainty_predictor'

# Legacy method (requires n_train >= 20)
'range_estimation_method': 'target_trajectory'
```

---

#### `range_scale_factor`

**Type:** `float`
**Default:** `3.0`
**Range:** `1.0` to `5.0` (typical)

For StandardScaler only: controls the range width as `[mean - k×scale, mean + k×scale]`.

**Common Values:**

| k | Coverage | Use Case |
|---|----------|----------|
| **2.0** | 95.45% | Narrower range, assumes inputs close to mean |
| **3.0** | 99.73% | **Recommended** - good balance |
| **4.0** | 99.99% | Conservative, very wide range |
| **5.0** | 99.9999% | Extremely conservative |

**Trade-offs:**
- **Lower k (2.0):**
  - Policy can focus on most common operating conditions
  - Risk: May clip extreme but valid inputs
  - Use when: Process always operates near mean conditions

- **Higher k (4.0+):**
  - Policy can explore full operational envelope
  - More conservative, handles rare conditions
  - Use when: Process has high variability

**Example:**
```python
# Standard coverage (recommended)
'range_scale_factor': 3.0

# Conservative (wider range)
'range_scale_factor': 4.0

# Focused (narrower range)
'range_scale_factor': 2.0
```

---

## Usage Examples

### Example 1: Standard Configuration (Recommended)

```python
'policy_generator': {
    'architecture': 'medium',
    'hidden_sizes': [64, 32],
    'dropout': 0.1,
    'use_batchnorm': False,

    # Normalization settings
    'range_estimation_method': 'uncertainty_predictor',
    'range_scale_factor': 3.0,
},
```

**Result:**
- Uses uncertainty predictor statistics
- Covers 99.7% of training data distribution
- Works with any number of scenarios

---

### Example 2: Conservative Configuration

```python
'policy_generator': {
    ...
    'range_estimation_method': 'uncertainty_predictor',
    'range_scale_factor': 4.0,  # More conservative
},
```

**Use when:**
- Process has high variability
- Safety-critical applications
- Want to explore full operational range

---

### Example 3: Legacy Configuration

```python
'scenarios': {
    'n_train': 50,  # Need many scenarios!
    ...
},
'policy_generator': {
    ...
    'range_estimation_method': 'target_trajectory',
    'range_scale_factor': 3.0,  # Ignored for target_trajectory
},
```

**Use when:**
- Need exact compatibility with old code
- Have many training scenarios (50+)

**WARNING:** Will fail with 1 scenario!

---

## How It Works

### StandardScaler Statistics

Uncertainty predictor training (5000 samples):

```
RF_Power: samples from [180, 400] W
Computed statistics:
  mean = 290.0 W
  scale (std dev) = 36.67 W
```

### Range Computation

With `range_scale_factor = 3.0`:

```
min = mean - 3 × scale = 290 - 3×36.67 = 180.0 W
max = mean + 3 × scale = 290 + 3×36.67 = 400.0 W
```

### Denormalization

Policy outputs normalized value:

```
normalized = 0.5 (from sigmoid)
denormalized = 180.0 + 0.5 × (400.0 - 180.0) = 290.0 W
```

---

## Debugging

### Verify Configuration

The configuration is printed during initialization:

```
Computing input ranges (method: uncertainty_predictor, scale_factor: 3.0):
  Input ranges for laser (uncertainty_predictor, k=3.0):
    Min: [0.15 20.0]
    Max: [1.50 40.0]
  Input ranges for plasma (uncertainty_predictor, k=3.0):
    Min: [180.0 15.0]
    Max: [400.0 45.0]
```

### Common Issues

**Issue: "Gradients are VANISHING"**
```
Solution: Check if using 1 scenario with 'target_trajectory' method
Fix: Switch to 'uncertainty_predictor' or increase n_train to 20+
```

**Issue: "Policy outputs out of expected range"**
```
Solution: Increase range_scale_factor from 3.0 to 4.0
```

**Issue: "BC loss is 0.0"**
```
Solution: Using 'target_trajectory' with 1 scenario
Fix: Switch to 'uncertainty_predictor'
```

---

## Mathematical Details

### For Uniform Distribution

If training data is uniformly distributed $U[a, b]$:

$$\mu = \frac{a+b}{2}, \quad \sigma = \frac{b-a}{2\sqrt{3}}$$

Range with k=3:

$$[\mu - 3\sigma, \mu + 3\sigma] = \left[\frac{a+b}{2} - \frac{\sqrt{3}}{2}(b-a), \frac{a+b}{2} + \frac{\sqrt{3}}{2}(b-a)\right]$$

Since $\frac{\sqrt{3}}{2} \approx 0.866$, this slightly underestimates [a, b] but covers 99.7% of samples.

### For Normal Distribution

If training data is $\mathcal{N}(\mu, \sigma^2)$:

- k=2: $\mu \pm 2\sigma$ covers 95.45%
- k=3: $\mu \pm 3\sigma$ covers 99.73%
- k=4: $\mu \pm 4\sigma$ covers 99.99%

---

## Best Practices

1. ✅ **Always use `'uncertainty_predictor'`** unless you have specific reasons not to

2. ✅ **Start with `range_scale_factor: 3.0`** - it works well for most cases

3. ✅ **Increase to 4.0** if process has high variability or rare operating conditions matter

4. ✅ **Use 2.0** only if you're confident the process operates near mean conditions

5. ❌ **Avoid `'target_trajectory'` method** - it's legacy and problematic

6. ✅ **Check the printed ranges** during initialization to verify they make physical sense

---

## Summary

| Setting | Recommended | Alternative | Legacy |
|---------|-------------|-------------|--------|
| **method** | `uncertainty_predictor` | - | `target_trajectory` |
| **scale_factor** | `3.0` | `4.0` (conservative) | N/A |
| **n_train** | Any (even 1) | 20+ (better) | 50+ (required for legacy) |

The recommended configuration works robustly in all scenarios and uses physically meaningful process ranges.
