"""
Test script to verify controllable inputs logic without full dependencies.
"""

# Simulated process configs
PROCESSES = [
    {
        'name': 'laser',
        'input_labels': ['PowerTarget', 'AmbientTemp'],
        'controllable_inputs': ['PowerTarget'],
    },
    {
        'name': 'plasma',
        'input_labels': ['RF_Power', 'Duration'],
        'controllable_inputs': ['RF_Power', 'Duration'],
    },
    {
        'name': 'microetch',
        'input_labels': ['Temperature', 'Concentration', 'Duration'],
        'controllable_inputs': ['Concentration', 'Duration'],
    }
]

def get_controllable_inputs(process_config):
    """Get controllable inputs with default fallback."""
    return process_config.get('controllable_inputs', process_config['input_labels'])


def simulate_constraint_application(process_config, generated_values, target_values):
    """
    Simulates the logic of _apply_non_controllable_constraints.

    Args:
        process_config: Process configuration
        generated_values: Dict of {input_name: generated_value}
        target_values: Dict of {input_name: target_value}

    Returns:
        Dict of final values after applying constraints
    """
    input_labels = process_config['input_labels']
    controllable = get_controllable_inputs(process_config)

    final_values = {}

    for label in input_labels:
        if label in controllable:
            # Use policy-generated value
            final_values[label] = generated_values[label]
        else:
            # Use target value (non-controllable)
            final_values[label] = target_values[label]

    return final_values


def test_microetch_scenario():
    """Test the microetch scenario where Temperature is non-controllable."""

    print("="*70)
    print("TEST: Microetch Process (Temperature non-controllable)")
    print("="*70)

    microetch_config = next(p for p in PROCESSES if p['name'] == 'microetch')

    # Scenario: Target has Temperature=300K
    target_values = {
        'Temperature': 300.0,
        'Concentration': 1.5,
        'Duration': 60.0
    }

    # Policy generator produces different values
    policy_generated = {
        'Temperature': 310.0,  # Will be OVERRIDDEN
        'Concentration': 2.0,  # Will be KEPT
        'Duration': 75.0       # Will be KEPT
    }

    # Apply constraints
    final_inputs = simulate_constraint_application(
        microetch_config, policy_generated, target_values
    )

    print(f"\nTarget values:           {target_values}")
    print(f"Policy generated:        {policy_generated}")
    print(f"Final inputs (after constraints): {final_inputs}")

    # Verify
    assert final_inputs['Temperature'] == 300.0, "Temperature should be from target"
    assert final_inputs['Concentration'] == 2.0, "Concentration should be from policy"
    assert final_inputs['Duration'] == 75.0, "Duration should be from policy"

    print("\n✓ TEST PASSED: Non-controllable constraints work correctly!")

    print("\nExplanation:")
    print("  - Temperature (non-controllable): inherited from target (300.0)")
    print("  - Concentration (controllable): from policy generator (2.0)")
    print("  - Duration (controllable): from policy generator (75.0)")


def test_laser_scenario():
    """Test the laser process where AmbientTemp is non-controllable."""

    print("\n" + "="*70)
    print("TEST: Laser Process (AmbientTemp non-controllable)")
    print("="*70)

    laser_config = next(p for p in PROCESSES if p['name'] == 'laser')

    target_values = {
        'PowerTarget': 0.8,
        'AmbientTemp': 25.0
    }

    policy_generated = {
        'PowerTarget': 0.9,   # Will be KEPT
        'AmbientTemp': 30.0   # Will be OVERRIDDEN
    }

    final_inputs = simulate_constraint_application(
        laser_config, policy_generated, target_values
    )

    print(f"\nTarget values:           {target_values}")
    print(f"Policy generated:        {policy_generated}")
    print(f"Final inputs (after constraints): {final_inputs}")

    assert final_inputs['PowerTarget'] == 0.9, "PowerTarget should be from policy"
    assert final_inputs['AmbientTemp'] == 25.0, "AmbientTemp should be from target"

    print("\n✓ TEST PASSED: Laser constraints work correctly!")


def test_plasma_scenario():
    """Test plasma process where all inputs are controllable."""

    print("\n" + "="*70)
    print("TEST: Plasma Process (All inputs controllable)")
    print("="*70)

    plasma_config = next(p for p in PROCESSES if p['name'] == 'plasma')

    target_values = {
        'RF_Power': 100.0,
        'Duration': 30.0
    }

    policy_generated = {
        'RF_Power': 120.0,   # Will be KEPT
        'Duration': 45.0     # Will be KEPT
    }

    final_inputs = simulate_constraint_application(
        plasma_config, policy_generated, target_values
    )

    print(f"\nTarget values:           {target_values}")
    print(f"Policy generated:        {policy_generated}")
    print(f"Final inputs (after constraints): {final_inputs}")

    assert final_inputs['RF_Power'] == 120.0, "RF_Power should be from policy"
    assert final_inputs['Duration'] == 45.0, "Duration should be from policy"

    print("\n✓ TEST PASSED: All inputs from policy (no constraints)!")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING CONTROLLABLE INPUTS CONSTRAINT LOGIC")
    print("="*70)

    test_microetch_scenario()
    test_laser_scenario()
    test_plasma_scenario()

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
