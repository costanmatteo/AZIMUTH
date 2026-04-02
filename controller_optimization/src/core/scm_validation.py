"""
SCM Validation Utilities

Validates that non-controllable inputs don't depend on controllable inputs
in the SCM causal graph. This prevents causal inconsistencies where:
- A variable is marked as non-controllable (environmental/fixed)
- But causally depends on a controllable variable in the SCM

Example of INVALID configuration:
  - Temperature is non-controllable (environmental)
  - But in SCM: Temperature = f(Concentration)
  - And Concentration IS controllable

This creates a logical contradiction.
"""

from configs.processes_config import get_controllable_inputs


class SCMValidationError(Exception):
    """Raised when SCM causal structure violates controllability constraints."""
    pass


def find_node_in_scm(scm_dataset, var_name):
    """
    Find a node specification in the SCM by variable name.

    Args:
        scm_dataset: SCMDataset instance
        var_name: Name of the variable to find

    Returns:
        NodeSpec if found, None otherwise
    """
    # SCMDataset.scm.specs is a Dict[str, NodeSpec]
    return scm_dataset.scm.specs.get(var_name, None)


def get_node_parents(scm_dataset, var_name):
    """
    Get the parent variables (direct causal dependencies) of a node.

    Args:
        scm_dataset: SCMDataset instance
        var_name: Name of the variable

    Returns:
        list: Parent variable names (empty if node not found or has no parents)
    """
    node = find_node_in_scm(scm_dataset, var_name)
    if node is None:
        return []
    return node.parents


def validate_non_controllable_independence(scm_dataset, process_config):
    """
    Validates that non-controllable inputs in the SCM don't causally depend
    on controllable inputs.

    This ensures logical consistency:
    - Non-controllable = environmental/fixed conditions
    - They cannot be influenced by variables the controller can modify

    Args:
        scm_dataset: SCMDataset instance
        process_config: Process configuration dict with 'input_labels'
                       and optionally 'controllable_inputs'

    Raises:
        SCMValidationError: If any non-controllable input depends on
                           a controllable input

    Example:
        >>> validate_non_controllable_independence(ds_scm_microetch, microetch_config)
        # Raises error if Temperature depends on Concentration
    """
    input_labels = process_config['input_labels']
    controllable = get_controllable_inputs(process_config)
    non_controllable = [x for x in input_labels if x not in controllable]

    errors = []

    for nc_var in non_controllable:
        # Get causal parents of this non-controllable variable
        parents = get_node_parents(scm_dataset, nc_var)

        # Check if any parent is a controllable input
        for parent in parents:
            if parent in controllable:
                errors.append(
                    f"Non-controllable input '{nc_var}' depends on "
                    f"controllable input '{parent}' in the SCM causal graph. "
                    f"This creates a logical inconsistency: a fixed environmental "
                    f"condition cannot depend on a controller-adjustable parameter."
                )

    if errors:
        error_msg = "\n\n".join([
            f"SCM VALIDATION FAILED for process '{process_config['name']}':",
            *[f"  - {err}" for err in errors],
            "",
            "Fix this by either:",
            "  1. Mark the parent variable as non-controllable, OR",
            "  2. Mark the dependent variable as controllable, OR",
            "  3. Restructure the SCM to remove the causal dependency"
        ])
        raise SCMValidationError(error_msg)


def validate_all_processes(processes_config_list):
    """
    Validates SCM consistency for all processes in the configuration.

    Args:
        processes_config_list: List of process configuration dicts

    Raises:
        SCMValidationError: If validation fails for any process
    """
    from controller_optimization.src.core.target_generation import get_scm_dataset

    print("Validating SCM causal consistency for all processes...")

    for process_config in processes_config_list:
        process_name = process_config['name']

        # Get SCM dataset
        scm_dataset = get_scm_dataset(process_config)

        # Validate
        try:
            validate_non_controllable_independence(scm_dataset, process_config)
            print(f"  ✓ {process_name}: SCM validation passed")
        except SCMValidationError as e:
            print(f"  ✗ {process_name}: SCM validation FAILED")
            raise e

    print("All processes validated successfully!\n")
