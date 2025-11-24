# Training Debugging Guide

## Overview
Extensive debugging has been added to trace every step of the training process. Debug mode is **automatically enabled for the first 3 epochs**.

## What Gets Logged

### 1. Process Chain Forward Pass (`process_chain.py`)
For each process in the chain:
- Initial inputs from target trajectory
- Scenario encoding (if enabled)
- Policy generator inputs/outputs
- Constraint application
- Input scaling/unscaling
- Uncertainty predictor outputs (mean, variance)
- Sampling with reparameterization trick

### 2. Loss Computation (`controller_trainer.py::compute_loss`)
- Reliability weight and lambda_bc values
- F (actual) vs F* (target) reliability
- Delta F and squared error
- Reliability loss (unscaled and scaled)
- Behavior cloning loss per process
- Per-process input comparisons (actual vs target, normalized)
- Final loss components breakdown

### 3. Gradient Analysis (`controller_trainer.py::train_epoch`)
For each policy generator:
- Total gradient norm
- Max/Min gradient values
- First 2 parameter gradients in detail
- **Warnings** if gradients are vanishing (<1e-10) or exploding (>100)

### 4. Parameter Updates (`controller_trainer.py::train_epoch`)
For each policy generator:
- Max parameter change after optimizer step
- Average parameter change
- First 2 parameter changes in detail
- **Warnings** if parameters barely change (<1e-10) or change dramatically (>0.1)

## How to Use

Simply run training normally:
```bash
python controller_optimization/train_controller.py
```

Debug output will automatically appear for **epochs 1, 2, and 3**.

## What to Look For

### ✅ Healthy Training Signs:
1. **Gradients**: In range 1e-5 to 1e-2
2. **Parameter changes**: In range 1e-6 to 1e-3
3. **Losses**: Decreasing over epochs
4. **F values**: Moving toward F*

### ⚠️ Problem Indicators:

#### Vanishing Gradients
```
Total gradient norm: 1.234567e-12  ⚠️  WARNING: Gradients are VANISHING!
```
**Causes:**
- Learning rate too low
- Reliability loss scale too small
- Gradients not flowing through chain

**Solutions:**
- Increase learning rate (1e-4 to 1e-3)
- Increase reliability_loss_scale
- Check surrogate weights balance

#### Exploding Gradients
```
Total gradient norm: 234.567890e+00  ⚠️  WARNING: Gradients are EXPLODING!
```
**Causes:**
- Learning rate too high
- Reliability loss scale too large
- Numerical instability

**Solutions:**
- Decrease learning rate
- Add gradient clipping
- Check for NaN/Inf values

#### Parameters Not Updating
```
Overall avg change: 1.234567e-11  ⚠️  WARNING: Parameters barely changed!
```
**Causes:**
- Learning rate too low
- Gradients vanishing
- Optimizer not working

**Solutions:**
- Increase learning rate significantly
- Check gradient flow
- Verify optimizer configuration

## Current Configuration Issues Found

Based on the visualization, the model is NOT training because:

### Issue 1: Learning Rate Too Low ⚠️
```python
# controller_config.py line 106
'learning_rate': 0.000001,  # 1e-6 is WAY too low!
```
**Recommended:** Change to `0.0001` (1e-4) or `0.001` (1e-3)

### Issue 2: Surrogate Weight Imbalance ⚠️
```python
# surrogate.py line 209-214
weights = {
    'laser': 0.1,      # Only 9% influence
    'plasma': 1.0,     # 91% influence
    'galvanic': 0.0,
    'microetch': 0.0
}
```
**Recommended:** Balance laser and plasma equally: `'laser': 1.0`

### Issue 3: Single Training Scenario ⚠️
```python
# controller_config.py line 146
'n_train': 1,  # Not really multi-scenario!
```
**Recommended:** Use at least 10-20 scenarios: `'n_train': 20`

## Diagnostic Script

A standalone diagnostic script is available:
```bash
python controller_optimization/debug_training.py
```

This tests:
- Gradient flow through the network
- Effect of different learning rates
- Loss component scales
- Parameter update magnitudes

## Files Modified

1. `controller_optimization/src/training/controller_trainer.py`
   - Added `debug` parameter to `compute_loss()`
   - Added `debug_first_scenario` parameter to `train_epoch()`
   - Modified `train()` to enable debug for first 3 epochs

2. `controller_optimization/src/utils/process_chain.py`
   - Added `debug` parameter to `forward()`
   - Added detailed logging at each step

3. `controller_optimization/debug_training.py`
   - NEW: Standalone diagnostic script

## Example Debug Output

```
======================================================================
DEBUG MODE ENABLED FOR EPOCH 1
======================================================================

######################################################################
TRAINING STEP DEBUG - First scenario in epoch
######################################################################
Scenario index: 0
Batch size: 64

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PROCESS CHAIN FORWARD DEBUG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Batch size: 64
Scenario idx: 0

Initial inputs (from target trajectory):
  Shape: torch.Size([64, 2])
  Values: [0.5 25.0]

--- Process 0: laser ---
  Scaled inputs: [-0.123 0.456]
  UncertPred outputs (unscaled):
    mean: [0.487]
    var:  [0.001]
  Sampled outputs:
    sampled: [0.491]

--- Process 1: plasma ---
  Policy input components:
    prev_outputs_mean: [0.491]
    prev_outputs_var:  [0.001]
  Generated inputs (raw): [198.5 29.8]
  Current inputs (after constraints): [198.5 29.8]
  ...

======================================================================
COMPUTE_LOSS DEBUG - Scenario 0
======================================================================
Reliability weight: 0.000000
Lambda BC: 10.000000

Reliability computation:
  F (actual):     0.123456
  F* (target):    0.987654
  Delta F:        -0.864198
  Reliability loss (scaled):   74.683951

Behavior Cloning computation:
  Process: laser
    Actual inputs:  [0.5 25.0]
    Target inputs:  [0.5 25.0]
    BC loss:        0.000000
  Process: plasma
    Actual inputs:  [198.5 29.8]
    Target inputs:  [200.0 30.0]
    BC loss:        0.012345

Final loss computation:
  Reliability component: 0.000000 × 74.683951 = 0.000000
  BC component:          10.000000 × 0.006172 = 0.061720
  Total loss:            0.061720

Gradient analysis:
  Policy generator 0:
    network.0.weight:
      Grad norm: 1.234567e-04
      Grad max:  2.345678e-05
    Total gradient norm: 3.456789e-04
    Max gradient:        2.345678e-05
    Min gradient:        1.234567e-08

Parameter update analysis:
  Policy generator 0:
    network.0.weight:
      Max change: 1.234567e-10  ⚠️  WARNING: Parameters barely changed!
      Avg change: 2.345678e-11
```

## Next Steps

1. Review debug output from epochs 1-3
2. Identify specific issue (gradients, learning rate, etc.)
3. Apply recommended fixes
4. Rerun training
5. Verify losses decrease and F values improve
