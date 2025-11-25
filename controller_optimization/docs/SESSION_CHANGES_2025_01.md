# Controller Optimization Session Changes - January 2025

This document summarizes all changes made during the debugging session to fix the PolicyGenerator training issues.

## Overview

**Problem**: The PolicyGenerator's RF_Power output was not changing during training, staying flat around 226-275 instead of learning to match the target value of ~336.

**Root Causes Found**:
1. Gradient flow was broken due to `.detach()` calls in scaling operations
2. Learning rate was set to 1e-6 (way too small)
3. BC loss normalization was suboptimal with single scenario

---

## Change 1: Bounded Policy Generator Outputs

### Files Modified
- `controller_optimization/src/models/policy_generator.py`
- `controller_optimization/src/utils/process_chain.py`
- `uncertainty_predictor/src/data/preprocessing.py`
- `controller_optimization/src/utils/model_utils.py`

### What Was Done
PolicyGenerator now outputs bounded values using tanh activation with affine scaling:

```python
def forward(self, x):
    features = self.network(x)
    raw_actions = self.output_head(features)

    if self.output_min is not None and self.output_max is not None:
        # Bounded output using tanh
        tanh_out = torch.tanh(raw_actions)
        normalized = 0.5 * (tanh_out + 1.0)  # Map to [0, 1]
        actions = self.output_min + normalized * (self.output_max - self.output_min)
    else:
        actions = raw_actions

    return actions
```

The bounds (min/max) are derived from the UncertaintyPredictor's training data via the `DataPreprocessor`:
- Added `input_min` and `input_max` tracking in `DataPreprocessor`
- ProcessChain passes these bounds to PolicyGenerators during construction

---

## Change 2: Differentiable Scaling Operations (CRITICAL FIX)

### Files Modified
- `controller_optimization/src/utils/process_chain.py`

### The Bug
The original `scale_inputs()` and `unscale_outputs()` methods used `.detach()` which **severed the computation graph**, preventing gradients from flowing back to the PolicyGenerator:

```python
# BEFORE (broken gradient flow)
def scale_inputs(self, inputs, process_idx):
    inputs_np = inputs.detach().cpu().numpy()  # <-- BREAKS GRADIENT FLOW!
    inputs_scaled = scaler.transform(inputs_np)
    return torch.tensor(inputs_scaled, ...)
```

### The Fix
Replaced numpy/sklearn operations with pure PyTorch operations:

```python
# AFTER (gradient flow preserved)
def scale_inputs(self, inputs, process_idx):
    scaler = self.preprocessors[process_idx].input_scaler
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=self.device)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=self.device)

    # Pure PyTorch - gradients flow through!
    inputs_scaled = (inputs - mean) / scale
    return inputs_scaled

def unscale_outputs(self, outputs, process_idx):
    scaler = self.preprocessors[process_idx].output_scaler
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=self.device)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=self.device)

    # Pure PyTorch - gradients flow through!
    outputs_unscaled = outputs * scale + mean
    return outputs_unscaled
```

### Why This Matters
The gradient chain should be:
```
Reliability Loss → Surrogate → UP outputs → UP inputs → PolicyGenerator outputs → PolicyGenerator weights
```

With `.detach()`, gradients stopped at the scaling step and never reached the PolicyGenerator.

---

## Change 3: Gradient Debugging

### Files Modified
- `controller_optimization/src/training/controller_trainer.py`
- `controller_optimization/train_controller.py`

### What Was Added
Added detailed gradient debugging after backward pass:

```python
# In train_epoch(), after total_loss.backward():
if hasattr(self, '_debug_gradients') and self._debug_gradients:
    print("GRADIENT DEBUG - After backward pass")

    # Check gradients on policy generators
    for i, policy in enumerate(self.process_chain.policy_generators):
        for name, param in policy.named_parameters():
            if param.grad is not None:
                print(f"  {name}: grad_norm={param.grad.norm()}")
            else:
                print(f"  {name}: NO GRADIENT!")

    # Check trajectory gradient functions
    for proc_name, data in trajectory.items():
        print(f"  {proc_name}:")
        print(f"    inputs.requires_grad: {data['inputs'].requires_grad}")
        print(f"    inputs.grad_fn: {data['inputs'].grad_fn}")
```

Enable with:
```python
trainer._debug_gradients = True
```

---

## Change 4: BC Loss Debugging

### Files Modified
- `controller_optimization/src/training/controller_trainer.py`
- `controller_optimization/train_controller.py`

### What Was Added
Added detailed BC loss debugging to see exactly what values are being compared:

```python
if hasattr(self, '_debug_bc_loss') and self._debug_bc_loss:
    print(f"BC Loss Debug [{process_name}]:")
    print(f"  actual_inputs (raw): {actual_inputs[0].tolist()}")
    print(f"  target_inputs (raw): {target_inputs_scenario[0].tolist()}")
    print(f"  stats['min']: {stats['min'].tolist()}")
    print(f"  stats['range']: {stats['range'].tolist()}")
    print(f"  actual_inputs_norm: {actual_inputs_norm[0].tolist()}")
    print(f"  target_inputs_norm: {target_inputs_norm[0].tolist()}")
    print(f"  MSE per dim: {((actual_inputs_norm - target_inputs_norm) ** 2).mean(dim=0).tolist()}")
```

Enable with:
```python
trainer._debug_bc_loss = True
```

---

## Change 5: ProcessChain Debug Mode

### Files Modified
- `controller_optimization/src/utils/process_chain.py`
- `controller_optimization/src/models/policy_generator.py`

### What Was Added
Class-level debug flag for verbose forward pass logging:

```python
class ProcessChain(nn.Module):
    debug = False  # Class-level flag

    @classmethod
    def enable_debug(cls, enable=True):
        cls.debug = enable
        PolicyGenerator.debug = enable
```

When enabled, prints step-by-step information:
- Policy generator inputs/outputs
- Scaled/unscaled values
- UncertaintyPredictor outputs
- Reparameterization sampling

---

## Issues Discovered But Not Fixed in Code

### 1. Learning Rate Too Small
**Location**: `controller_optimization/configs/controller_config.py`

```python
'learning_rate': 0.000001,  # 1e-6 is WAY TOO SMALL!
```

**Recommendation**: Increase to `0.001` or `0.0001`

### 2. BC Loss Normalization with Single Scenario
**Location**: `controller_optimization/src/training/controller_trainer.py`

With `n_train=1`, the normalization stats become degenerate:
- `input_min = input_max = single_target_value`
- `input_range ≈ 1e-8` (just epsilon)

This causes the normalization to use a single global min/max across all input dimensions instead of per-dimension normalization.

**Recommendation**: Use `n_train >= 5` scenarios for proper normalization

### 3. Surrogate Loss Direction
The surrogate computes reliability F, and the loss is `(F - F*)²`. When F > F* (controller produces better quality than target), the loss pushes F DOWN instead of rewarding the improvement.

**Recommendation**: Consider asymmetric loss that only penalizes F < F*, or maximize F directly.

---

## Verification Checklist

After making these changes, verify:

1. **Gradients flow to PolicyGenerator**:
   ```
   Policy Generator 0:
     Total grad norm: 0.8893...  (should be > 0)
   ```

2. **Plasma inputs have gradient functions**:
   ```
   plasma:
     inputs.requires_grad: True
     inputs.grad_fn: <CopySlices object>  (should NOT be None)
   ```

3. **BC loss sees the gap**:
   ```
   BC Loss Debug [plasma]:
     actual_inputs (raw): [247.15, 51.68]
     target_inputs (raw): [336.23, 51.68]
     MSE per dim: [0.099, 0.0]  (should be > 0 for RF_Power)
   ```

4. **RF_Power changes during training**: Check the training progression plot

---

## Files Changed Summary

| File | Change Type |
|------|-------------|
| `process_chain.py` | Fixed gradient flow in scaling, added debug mode |
| `policy_generator.py` | Added bounded output, debug mode |
| `preprocessing.py` | Added input_min/max tracking |
| `model_utils.py` | Added backward compatibility for input bounds |
| `controller_trainer.py` | Added gradient and BC loss debugging |
| `train_controller.py` | Enable debug flags |
| `processes_config.py` | Fixed controllable_inputs for plasma (reverted incorrect change) |

---

## Commits Made

1. `Fix: Make scaling operations differentiable for gradient flow`
2. `Add gradient debugging to trace where gradients break`
3. `Add BC loss debugging to trace normalization issue`
