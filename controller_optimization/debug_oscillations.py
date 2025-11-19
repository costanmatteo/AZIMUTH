"""
Script di debug per capire perché la loss oscilla.

Testa:
1. Stocasticità del sampling (reparameterization trick)
2. Varianze predette dagli UncertaintyPredictors
3. Gradient flow
4. Loss landscape
"""

import sys
from pathlib import Path
import torch
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.configs.processes_config import PROCESSES
from controller_optimization.configs.controller_config import CONTROLLER_CONFIG
from controller_optimization.src.utils.target_generation import generate_target_trajectory
from controller_optimization.src.utils.process_chain import ProcessChain
from controller_optimization.src.models.surrogate import ProTSurrogate

print("="*70)
print("DEBUG: Oscillazioni Loss")
print("="*70)

# Setup
device = 'cpu'
n_train = 1

# Usa solo laser e plasma
PROCESSES_SUBSET = [p for p in PROCESSES if p['name'] in ['laser', 'plasma']]
print(f"\nProcesses: {[p['name'] for p in PROCESSES_SUBSET]}")

# Generate target
target_trajectory = generate_target_trajectory(
    process_configs=PROCESSES_SUBSET,
    n_samples=n_train,
    seed=42
)

# Create chain and surrogate
chain = ProcessChain(
    processes_config=PROCESSES_SUBSET,
    target_trajectory=target_trajectory,
    policy_config=CONTROLLER_CONFIG['policy_generator'],
    device=device
)

# Test with stochastic sampling to see the oscillations
surrogate = ProTSurrogate(target_trajectory, device=device, use_deterministic_sampling=False)

print(f"\nPolicy networks: {len(chain.policy_generators)}")
print(f"F_star: {surrogate.F_star[0]:.6f}")

# ============================================================================
# TEST 1: Stocasticità del sampling
# ============================================================================
print("\n" + "="*70)
print("TEST 1: Stocasticità del Sampling")
print("="*70)
print("Eseguo 10 forward pass con STESSI PESI e misuro varianza della loss")

chain.eval()
losses = []
F_values = []

with torch.no_grad():
    for i in range(10):
        trajectory = chain.forward(batch_size=1, scenario_idx=0)
        F = surrogate.compute_reliability(trajectory).item()
        loss = (F - surrogate.F_star[0]) ** 2

        losses.append(loss)
        F_values.append(F)
        print(f"  Run {i+1}: F={F:.6f}, loss={loss:.6f}")

losses = np.array(losses)
F_values = np.array(F_values)

print(f"\nStatistiche (stessi pesi, 10 runs):")
print(f"  F mean: {F_values.mean():.6f}, std: {F_values.std():.6f}")
print(f"  Loss mean: {losses.mean():.6f}, std: {losses.std():.6f}")
print(f"  Coefficiente variazione loss: {losses.std() / losses.mean() * 100:.1f}%")

if losses.std() / losses.mean() > 0.1:
    print("\n⚠️  PROBLEMA: Loss varia >10% con stessi pesi!")
    print("   Causa: Sampling stocastico nel reparameterization trick")
    print("   Soluzione: Usa outputs_mean direttamente (no sampling)")

# ============================================================================
# TEST 2: Varianze predette
# ============================================================================
print("\n" + "="*70)
print("TEST 2: Varianze Predette dagli UncertaintyPredictors")
print("="*70)

with torch.no_grad():
    trajectory = chain.forward(batch_size=1, scenario_idx=0)

    for process_name, data in trajectory.items():
        mean = data['outputs_mean'].squeeze()
        var = data['outputs_var'].squeeze()
        std = torch.sqrt(var)

        print(f"\n{process_name}:")
        print(f"  Output mean: {mean.item():.6f}")
        print(f"  Output var: {var.item():.6f}")
        print(f"  Output std: {std.item():.6f}")
        print(f"  Signal-to-noise (mean/std): {abs(mean.item() / std.item()):.2f}")

        if abs(mean.item() / std.item()) < 3.0:
            print(f"  ⚠️  PROBLEMA: SNR < 3 (alta varianza rispetto al segnale)")

# ============================================================================
# TEST 3: Gradient flow
# ============================================================================
print("\n" + "="*70)
print("TEST 3: Gradient Flow")
print("="*70)

chain.train()

# Forward
trajectory = chain.forward(batch_size=1, scenario_idx=0)
F = surrogate.compute_reliability(trajectory)
loss = (F - surrogate.F_star[0]) ** 2

# Backward
chain.zero_grad()
loss.backward()

print(f"\nF: {F.item():.6f}, Loss: {loss.item():.6f}")
print("\nGradienti:")

has_gradients = False
for name, param in chain.named_parameters():
    if param.requires_grad:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            print(f"  {name}:")
            print(f"    grad_norm: {grad_norm:.6f}")
            print(f"    param_norm: {param_norm:.6f}")
            print(f"    ratio: {grad_norm/param_norm:.6f}")
            has_gradients = True

            if grad_norm < 1e-6:
                print(f"    ⚠️  Gradiente quasi zero!")
            elif grad_norm > 100:
                print(f"    ⚠️  Gradiente molto alto!")

if not has_gradients:
    print("  ❌ PROBLEMA: Nessun gradiente trovato!")

# ============================================================================
# TEST 4: Loss landscape (step lungo gradiente)
# ============================================================================
print("\n" + "="*70)
print("TEST 4: Loss Landscape")
print("="*70)
print("Faccio piccoli step lungo il gradiente e misuro come cambia la loss")

chain.eval()

# Salva stato originale
original_state = {name: param.clone() for name, param in chain.named_parameters() if param.requires_grad}

# Misura loss originale (media su 5 forward per ridurre rumore)
original_losses = []
with torch.no_grad():
    for _ in range(5):
        traj = chain.forward(batch_size=1, scenario_idx=0)
        F = surrogate.compute_reliability(traj)
        original_losses.append((F - surrogate.F_star[0]) ** 2)
original_loss = np.mean(original_losses)

print(f"\nLoss originale (avg 5 runs): {original_loss:.6f}")

# Fai piccoli step lungo il gradiente
step_sizes = [0.0001, 0.001, 0.01, 0.1]
for step_size in step_sizes:
    # Restore e applica step
    for name, param in chain.named_parameters():
        if param.requires_grad:
            param.data = original_state[name] - step_size * param.grad

    # Misura nuova loss (media su 5)
    new_losses = []
    with torch.no_grad():
        for _ in range(5):
            traj = chain.forward(batch_size=1, scenario_idx=0)
            F = surrogate.compute_reliability(traj)
            new_losses.append((F - surrogate.F_star[0]) ** 2)
    new_loss = np.mean(new_losses)

    delta = original_loss - new_loss
    print(f"  Step size {step_size:.4f}: loss={new_loss:.6f}, delta={delta:.6f}")

    if delta > 0:
        print(f"    ✓ Loss migliorata!")
    elif delta < -0.001:
        print(f"    ⚠️  Loss peggiorata (landscape difficile o LR troppo alto)")

# Restore
for name, param in chain.named_parameters():
    if param.requires_grad:
        param.data = original_state[name]

# ============================================================================
# TEST 5: Loss con sampling vs senza sampling
# ============================================================================
print("\n" + "="*70)
print("TEST 5: Loss con Sampling vs Senza Sampling")
print("="*70)

# Create deterministic surrogate using the new parameter
surrogate_det = ProTSurrogate(target_trajectory, device=device, use_deterministic_sampling=True)

print("\nCon sampling (10 runs):")
losses_sampling = []
with torch.no_grad():
    for _ in range(10):
        traj = chain.forward(batch_size=1, scenario_idx=0)
        F = surrogate.compute_reliability(traj)
        losses_sampling.append((F - surrogate.F_star[0]) ** 2)
print(f"  Mean: {np.mean(losses_sampling):.6f}, Std: {np.std(losses_sampling):.6f}")

print("\nSenza sampling (10 runs):")
losses_no_sampling = []
with torch.no_grad():
    for _ in range(10):
        traj = chain.forward(batch_size=1, scenario_idx=0)
        F = surrogate_det.compute_reliability(traj)
        losses_no_sampling.append((F - surrogate_det.F_star[0]) ** 2)
print(f"  Mean: {np.mean(losses_no_sampling):.6f}, Std: {np.std(losses_no_sampling):.6f}")

variance_reduction = (1 - np.std(losses_no_sampling) / np.std(losses_sampling)) * 100
print(f"\nRiduzione varianza senza sampling: {variance_reduction:.1f}%")

if variance_reduction > 50:
    print("✓ SOLUZIONE: Rimuovi sampling per ridurre oscillazioni!")

print("\n" + "="*70)
print("RACCOMANDAZIONI")
print("="*70)

recommendations = []

if losses.std() / losses.mean() > 0.1:
    recommendations.append("1. Rimuovi sampling stocastico (usa outputs_mean direttamente)")

if variance_reduction > 50:
    recommendations.append("2. Implementa surrogate deterministico")

# Check if any gradient is too small
min_grad = float('inf')
for name, param in chain.named_parameters():
    if param.requires_grad and param.grad is not None:
        min_grad = min(min_grad, param.grad.abs().max().item())

if min_grad < 1e-5:
    recommendations.append("3. Aumenta reliability_loss_scale (es. 1000.0)")

recommendations.append("4. Aumenta batch_size (es. 256) per gradienti più stabili")
recommendations.append("5. Riduci learning_rate (es. 0.00001)")

for rec in recommendations:
    print(f"  {rec}")

print("\n" + "="*70)
