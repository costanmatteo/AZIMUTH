"""
Debug dettagliato del surrogato SCM - trova esattamente dove si generano NaN.
"""

import sys
from pathlib import Path
import numpy as np

# Add paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.configs.processes_config import PROCESSES

def debug_scm_surrogate_numpy(process_config):
    """
    Simula il surrogato SCM usando NumPy puro per debug.
    Stampa ogni singolo step per identificare dove si generano NaN.
    """
    process_name = process_config['name']
    print(f"\n{'='*70}")
    print(f"DEBUG {process_name.upper()} - Usando NumPy puro")
    print(f"{'='*70}")

    # Import dataset
    from uncertainty_predictor.scm_ds.datasets import (
        ds_scm_laser,
        ds_scm_plasma,
        ds_scm_galvanic,
        ds_scm_microetch
    )

    dataset_map = {
        'laser': ds_scm_laser,
        'plasma': ds_scm_plasma,
        'galvanic': ds_scm_galvanic,
        'microetch': ds_scm_microetch
    }

    dataset = dataset_map[process_name]
    scm = dataset.scm

    # Crea input di test
    if process_name == 'laser':
        inputs = np.array([[0.5, 25.0]])  # PowerTarget, AmbientTemp
    elif process_name == 'plasma':
        inputs = np.array([[200.0, 30.0]])  # RF_Power, Duration
    elif process_name == 'galvanic':
        inputs = np.array([[3.0, 1800.0]])  # CurrentDensity, Duration
    elif process_name == 'microetch':
        inputs = np.array([[298.0, 1.5, 60.0]])  # Temperature, Concentration, Duration
    else:
        print(f"Unknown process: {process_name}")
        return

    print(f"\nInput values:")
    for i, label in enumerate(process_config['input_labels']):
        print(f"  {label:20s} = {inputs[0, i]:.4f}")

    # Genera sample dal dataset SCM con rumore a ZERO
    print(f"\nGenerating sample with ZERO noise...")

    # Crea context
    context = {}

    # Step 1: Aggiungi input
    print(f"\n--- Step 1: Input Variables ---")
    for i, label in enumerate(process_config['input_labels']):
        context[label] = inputs[:, i]
        print(f"  {label:20s} = {context[label][0]:.6f}")

    # Step 2: Genera noise (settato a zero per deterministico)
    print(f"\n--- Step 2: Noise Terms (set to zero) ---")
    rng = np.random.default_rng(42)
    eps_draws = {}

    for node_name in scm.order:
        eps_name = f"eps_{node_name}"

        # Check if this node has a sampler (constant values)
        if node_name in dataset.noise_model.singles:
            sampler = dataset.noise_model.singles[node_name]
            value = sampler(rng, 1)[0]
            eps_draws[node_name] = np.array([value])

            # Se è una costante, stampala
            if node_name not in process_config['input_labels'] and node_name not in process_config['output_labels']:
                print(f"  {node_name:20s} (const) = {value:.6f}")
        else:
            # Regular noise: set to zero
            eps_draws[node_name] = np.array([0.0])

    # Step 3: Forward pass attraverso i nodi
    print(f"\n--- Step 3: Node Evaluation (Topological Order) ---")

    for node_name in scm.order:
        if node_name in context:
            # Already computed (input)
            continue

        spec = scm.specs[node_name]
        parents = spec.parents

        # Get parent values
        parent_vals = [context[p] for p in parents] + [eps_draws[node_name]]

        # Evaluate function
        try:
            result = scm._fns[node_name](*parent_vals)
            context[node_name] = result

            # Check for NaN/Inf
            has_nan = np.isnan(result).any()
            has_inf = np.isinf(result).any()

            status = ""
            if has_nan:
                status = " ← ⚠️ NaN DETECTED!"
            elif has_inf:
                status = " ← ⚠️ Inf DETECTED!"

            print(f"  {node_name:20s} = {result[0]:.6f}{status}")

            # Se c'è NaN, stampa i dettagli
            if has_nan or has_inf:
                print(f"    Expression: {spec.expr}")
                print(f"    Parents: {parents}")
                for p in parents:
                    print(f"      {p} = {context[p][0]:.6f}")
                print(f"      eps_{node_name} = {eps_draws[node_name][0]:.6f}")

                # STOP al primo NaN
                print(f"\n{'='*70}")
                print(f"NaN/Inf FOUND IN NODE: {node_name}")
                print(f"{'='*70}")
                return False

        except Exception as e:
            print(f"  {node_name:20s} ← ERROR: {e}")
            print(f"    Expression: {spec.expr}")
            print(f"    Parents: {parents}")
            return False

    # Step 4: Check output
    print(f"\n--- Step 4: Output Variables ---")
    for label in process_config['output_labels']:
        value = context[label][0]
        has_nan = np.isnan(value)
        has_inf = np.isinf(value)

        status = ""
        if has_nan:
            status = " ← ⚠️ NaN!"
        elif has_inf:
            status = " ← ⚠️ Inf!"

        print(f"  {label:20s} = {value:.6f}{status}")

        if has_nan or has_inf:
            return False

    print(f"\n✓ No NaN/Inf detected in {process_name}!")
    return True


if __name__ == '__main__':
    print("="*70)
    print("SCM SURROGATE NaN DEBUG")
    print("="*70)

    # Test processes
    process_names = ['laser', 'plasma']

    all_passed = True
    for process_config in PROCESSES:
        if process_config['name'] in process_names:
            passed = debug_scm_surrogate_numpy(process_config)
            all_passed = all_passed and passed

            if not passed:
                print(f"\n⚠️ Stopping at first failure in {process_config['name']}")
                break

    print(f"\n{'='*70}")
    if all_passed:
        print("✓ ALL TESTS PASSED - NO NaN/Inf!")
    else:
        print("✗ FOUND NaN/Inf - See details above")
    print(f"{'='*70}")
