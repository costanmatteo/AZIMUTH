"""
Test script per verificare il filtraggio dei processi basato su CONTROLLER_CONFIG.

Questo script dimostra:
1. get_filtered_processes() filtra correttamente i processi
2. L'ordine è mantenuto come specificato in process_names
3. Validation degli errori se un processo non esiste
"""

import sys
from pathlib import Path

# Add paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from controller_optimization.configs.processes_config import (
    PROCESSES,
    get_filtered_processes
)
from controller_optimization.configs.controller_config import CONTROLLER_CONFIG


def test_process_filtering():
    """Test che il filtraggio dei processi funzioni correttamente."""

    print("=" * 80)
    print("TEST: Process Filtering")
    print("=" * 80)

    # Test 1: Tutti i processi (None)
    print("\n1. Test con process_names=None (tutti i processi):")
    all_procs = get_filtered_processes(None)
    print(f"   Input: None")
    print(f"   Output: {[p['name'] for p in all_procs]}")
    assert len(all_procs) == len(PROCESSES), "Should return all processes"
    print("   ✓ PASS: Ritorna tutti i processi")

    # Test 2: Solo 2 processi
    print("\n2. Test con process_names=['laser', 'plasma']:")
    filtered_2 = get_filtered_processes(['laser', 'plasma'])
    filtered_names = [p['name'] for p in filtered_2]
    print(f"   Input: ['laser', 'plasma']")
    print(f"   Output: {filtered_names}")
    assert filtered_names == ['laser', 'plasma'], "Should return only specified processes"
    assert len(filtered_2) == 2, "Should return 2 processes"
    print("   ✓ PASS: Filtra correttamente")

    # Test 3: Ordine personalizzato
    print("\n3. Test con ordine personalizzato ['plasma', 'laser']:")
    filtered_rev = get_filtered_processes(['plasma', 'laser'])
    filtered_names_rev = [p['name'] for p in filtered_rev]
    print(f"   Input: ['plasma', 'laser']")
    print(f"   Output: {filtered_names_rev}")
    assert filtered_names_rev == ['plasma', 'laser'], "Should respect specified order"
    print("   ✓ PASS: Ordine mantenuto")

    # Test 4: Tutti e 4 i processi
    print("\n4. Test con tutti i 4 processi:")
    all_4 = get_filtered_processes(['laser', 'plasma', 'galvanic', 'microetch'])
    all_4_names = [p['name'] for p in all_4]
    print(f"   Input: ['laser', 'plasma', 'galvanic', 'microetch']")
    print(f"   Output: {all_4_names}")
    assert len(all_4) == 4, "Should return 4 processes"
    print("   ✓ PASS: Tutti e 4 i processi")

    # Test 5: Processo non esistente (dovrebbe dare errore)
    print("\n5. Test con processo non esistente ['laser', 'invalid']:")
    try:
        get_filtered_processes(['laser', 'invalid'])
        print("   ✗ FAIL: Dovrebbe lanciare ValueError")
        return False
    except ValueError as e:
        print(f"   ✓ PASS: ValueError lanciato correttamente")
        print(f"   Messaggio: {e}")

    # Test 6: Verifica con CONTROLLER_CONFIG
    print("\n6. Test con CONTROLLER_CONFIG['process_names']:")
    config_process_names = CONTROLLER_CONFIG.get('process_names', None)
    print(f"   CONTROLLER_CONFIG['process_names'] = {config_process_names}")

    if config_process_names is not None:
        config_filtered = get_filtered_processes(config_process_names)
        config_names = [p['name'] for p in config_filtered]
        print(f"   Processi filtrati: {config_names}")
        assert config_names == config_process_names, "Should match config"
        print("   ✓ PASS: Config correttamente applicata")
    else:
        print("   ℹ process_names = None, usando tutti i processi")
        config_filtered = get_filtered_processes(None)
        print(f"   Processi: {[p['name'] for p in config_filtered]}")

    # Test 7: Verifica struttura dei dati
    print("\n7. Test struttura dati:")
    filtered = get_filtered_processes(['laser', 'plasma'])
    laser_config = filtered[0]
    print(f"   Processo: {laser_config['name']}")
    print(f"   Keys: {list(laser_config.keys())}")
    assert 'name' in laser_config, "Should have 'name' field"
    assert 'input_dim' in laser_config, "Should have 'input_dim' field"
    assert 'output_dim' in laser_config, "Should have 'output_dim' field"
    assert 'checkpoint_dir' in laser_config, "Should have 'checkpoint_dir' field"
    print("   ✓ PASS: Struttura dati completa")

    print("\n" + "=" * 80)
    print("✓ TUTTI I TEST PASSATI!")
    print("=" * 80)

    print("\nRiepilogo funzionalità:")
    print("  1. get_filtered_processes(None) → tutti i processi di PROCESSES")
    print("  2. get_filtered_processes(['a', 'b']) → solo processi 'a' e 'b'")
    print("  3. L'ordine dei processi è quello specificato in input")
    print("  4. Validation: errore se un processo non esiste")
    print("  5. Integrato con CONTROLLER_CONFIG['process_names']")

    return True


if __name__ == '__main__':
    success = test_process_filtering()
    sys.exit(0 if success else 1)
