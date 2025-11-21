"""
SCM Surrogate - Surrogato deterministico basato su funzioni strutturali causali.

Invece di usare una rete neurale (UncertaintyPredictor) per predire gli output
dei macchinari, questo surrogato usa direttamente le funzioni strutturali causali
(SCM) definite nel dataset.

Vantaggi:
- Deterministico e interpretabile
- Perfettamente allineato con il data generation process
- Differenziabile (gradiente può fluire)
- No training necessario

Implementazione:
- Compila le funzioni SCM con PyTorch (backend differenziabile)
- Implementa lo stesso interface di UncertaintyPredictor
- Setta il rumore a zero per ottenere le funzioni deterministiche
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from typing import List, Dict, Callable

# Add paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from uncertainty_predictor.scm_ds.scm import NodeSpec


class SCMProcessSurrogate(nn.Module):
    """
    Surrogato deterministico per un singolo processo basato su funzioni SCM.

    Sostituisce UncertaintyPredictor usando le funzioni strutturali causali
    definite nel dataset invece di una rete neurale.

    Args:
        node_specs (List[NodeSpec]): Specifiche dei nodi SCM per questo processo
        input_labels (List[str]): Nomi delle variabili di input
        output_labels (List[str]): Nomi delle variabili di output
        min_variance (float): Varianza minima da ritornare (per compatibilità)
        device (str): Device PyTorch
    """

    def __init__(
        self,
        node_specs: List[NodeSpec],
        input_labels: List[str],
        output_labels: List[str],
        noise_samplers: Dict[str, Callable] = None,
        min_variance: float = 1e-6,
        device: str = 'cpu'
    ):
        super(SCMProcessSurrogate, self).__init__()

        self.device = device
        self.input_labels = input_labels
        self.output_labels = output_labels
        self.min_variance = min_variance

        # Build node specs dictionary
        self.specs: Dict[str, NodeSpec] = {s.name: s for s in node_specs}

        # Store noise samplers for constant nodes
        self.noise_samplers = noise_samplers or {}

        # Compute topological order
        self.order = self._topo_order()

        # Compile functions using PyTorch
        self._compile_pytorch_functions()

    def _topo_order(self) -> List[str]:
        """
        Compute topological order of the DAG.
        Same as SCM._topo_order() but simplified.
        """
        from collections import defaultdict, deque

        indeg: Dict[str, int] = defaultdict(int)
        graph: Dict[str, List[str]] = defaultdict(list)

        # Build adjacency and indegree
        for s in self.specs.values():
            for p in s.parents:
                graph[p].append(s.name)
                indeg[s.name] += 1
            indeg.setdefault(s.name, 0)

        # Kahn's algorithm
        q: deque = deque([v for v, d in indeg.items() if d == 0])
        order: List[str] = []
        while q:
            u = q.popleft()
            order.append(u)
            for w in graph[u]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)

        if len(order) != len(self.specs):
            raise ValueError("Graph has a cycle; SCM requires a DAG.")
        return order

    def _compile_pytorch_functions(self):
        """
        Compile SCM expressions to PyTorch functions.

        We manually implement each process's equations using PyTorch operations
        to ensure differentiability.
        """
        # Note: SymPy lambdify with 'torch' backend has limited support
        # and doesn't work well with all operations (exp, sqrt, etc.)
        #
        # Instead, we manually implement the equations for each process
        # using PyTorch operations. This is more reliable and faster.

        # We'll store the compiled functions as Python callables
        self._fns: Dict[str, Callable] = {}

        for v in self.order:
            spec = self.specs[v]
            # We'll compile this on-demand in forward() using torch operations
            # For now, store the spec
            self._fns[v] = spec

    def _safe_exp(self, x):
        """Safe exponential with clipping to prevent overflow."""
        return torch.exp(torch.clamp(x, min=-20, max=20))

    def _safe_sqrt(self, x):
        """Safe square root with absolute value."""
        return torch.sqrt(torch.abs(x) + 1e-8)

    def _evaluate_expression(self, node_name: str, context: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate a node's expression using PyTorch operations.

        Args:
            node_name: Name of the node to evaluate
            context: Dictionary mapping variable names to their tensor values

        Returns:
            Evaluated tensor
        """
        spec = self.specs[node_name]

        # Build symbol dictionary for SymPy parsing
        symbols_dict = {}
        for parent in spec.parents:
            symbols_dict[parent] = sp.Symbol(parent)

        # Add noise symbol (we'll set it to zero for deterministic evaluation)
        eps_name = f"eps_{node_name}"
        symbols_dict[eps_name] = sp.Symbol(eps_name)

        # Parse expression
        expr = sp.sympify(spec.expr, locals=symbols_dict)

        # Manually convert SymPy expression to PyTorch operations
        # This is a simplified converter - handles common operations
        result = self._sympy_to_torch(expr, context, node_name)

        return result

    def _sympy_to_torch(self, expr, context: Dict[str, torch.Tensor], node_name: str) -> torch.Tensor:
        """
        Convert SymPy expression to PyTorch tensor operations.

        This is a recursive converter that handles common operations.
        """
        # Check if it's a symbol (variable)
        if isinstance(expr, sp.Symbol):
            var_name = str(expr)
            # Handle noise terms
            if var_name.startswith('eps_'):
                # Extract node name from eps_<name>
                node_from_eps = var_name[4:]  # Remove 'eps_' prefix
                batch_size = next(iter(context.values())).shape[0]

                # Check if we have a sampler for this node (constant nodes)
                if node_from_eps in self.noise_samplers:
                    # Use the sampler to generate constant values
                    # Create a dummy RNG (values are deterministic anyway)
                    import numpy as np
                    dummy_rng = np.random.default_rng(42)
                    values = self.noise_samplers[node_from_eps](dummy_rng, batch_size)
                    return torch.tensor(values, dtype=torch.float32, device=self.device)
                else:
                    # Regular noise term: set to zero for deterministic evaluation
                    return torch.zeros(batch_size, device=self.device)
            # Return variable from context
            if var_name in context:
                return context[var_name]
            else:
                raise ValueError(f"Variable {var_name} not found in context")

        # Check if it's a number
        if isinstance(expr, (sp.Integer, sp.Float, sp.Rational)):
            batch_size = next(iter(context.values())).shape[0]
            return torch.full((batch_size,), float(expr), device=self.device)

        # Handle operations
        if isinstance(expr, sp.Add):
            # Addition: sum all terms
            result = None
            for arg in expr.args:
                term = self._sympy_to_torch(arg, context, node_name)
                if result is None:
                    result = term
                else:
                    result = result + term
            return result

        elif isinstance(expr, sp.Mul):
            # Multiplication: multiply all terms
            result = None
            for arg in expr.args:
                term = self._sympy_to_torch(arg, context, node_name)
                if result is None:
                    result = term
                else:
                    result = result * term
            return result

        elif isinstance(expr, sp.Pow):
            # Power: base ** exponent
            base = self._sympy_to_torch(expr.base, context, node_name)
            exp = self._sympy_to_torch(expr.exp, context, node_name)

            # Handle special cases
            if isinstance(expr.exp, (sp.Integer, sp.Float, sp.Rational)):
                exp_val = float(expr.exp)
                if exp_val == 0.5:
                    return self._safe_sqrt(base)
                elif exp_val == 2:
                    return base * base

            return torch.pow(base, exp)

        elif isinstance(expr, sp.exp):
            # Exponential
            arg = self._sympy_to_torch(expr.args[0], context, node_name)
            return self._safe_exp(arg)

        elif isinstance(expr, sp.log):
            # Logarithm
            arg = self._sympy_to_torch(expr.args[0], context, node_name)
            return torch.log(torch.abs(arg) + 1e-8)

        elif isinstance(expr, sp.sin):
            # Sine
            arg = self._sympy_to_torch(expr.args[0], context, node_name)
            return torch.sin(arg)

        elif isinstance(expr, sp.cos):
            # Cosine
            arg = self._sympy_to_torch(expr.args[0], context, node_name)
            return torch.cos(arg)

        elif isinstance(expr, sp.Abs):
            # Absolute value
            arg = self._sympy_to_torch(expr.args[0], context, node_name)
            return torch.abs(arg)

        else:
            raise NotImplementedError(f"Unsupported SymPy operation: {type(expr)} in {expr}")

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass usando le funzioni SCM deterministiche.

        Args:
            x: Input tensor, shape (batch_size, input_dim)
               Columns correspond to self.input_labels

        Returns:
            tuple: (mean, variance)
                - mean: Output tensor, shape (batch_size, output_dim)
                - variance: Constant small variance, shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Build context dictionary from inputs
        context: Dict[str, torch.Tensor] = {}
        for i, label in enumerate(self.input_labels):
            # Extract column i from input tensor
            context[label] = x[:, i]

        # Evaluate nodes in topological order
        for node_name in self.order:
            # Skip if already computed (from inputs)
            if node_name not in context:
                # Evaluate this node
                context[node_name] = self._evaluate_expression(node_name, context)

        # Extract outputs
        outputs = []
        for label in self.output_labels:
            if label not in context:
                raise ValueError(f"Output variable {label} not computed in SCM forward pass")
            outputs.append(context[label].unsqueeze(1))  # Add feature dimension

        mean = torch.cat(outputs, dim=1)  # Shape: (batch_size, output_dim)

        # Return zero variance (deterministic model)
        variance = torch.full_like(mean, self.min_variance)

        return mean, variance


def create_scm_surrogate_for_process(process_config: dict, device: str = 'cpu') -> SCMProcessSurrogate:
    """
    Factory function to create an SCM surrogate for a specific process.

    Args:
        process_config: Process configuration from processes_config.py
        device: PyTorch device

    Returns:
        SCMProcessSurrogate instance
    """
    # Get SCM dataset for this process
    process_name = process_config['name']

    # Import the specific dataset
    from uncertainty_predictor.scm_ds.datasets import (
        ds_scm_laser,
        ds_scm_plasma,
        ds_scm_galvanic,
        ds_scm_microetch
    )

    # Map process name to dataset
    dataset_map = {
        'laser': ds_scm_laser,
        'plasma': ds_scm_plasma,
        'galvanic': ds_scm_galvanic,
        'microetch': ds_scm_microetch
    }

    if process_name not in dataset_map:
        raise ValueError(f"Unknown process: {process_name}")

    dataset = dataset_map[process_name]

    # Extract node specs from the dataset
    node_specs = list(dataset.scm.specs.values())

    # Extract noise samplers for constant nodes
    # The noise_model contains singles (individual samplers) and groups (correlated samplers)
    noise_samplers = {}
    if hasattr(dataset, 'noise_model') and dataset.noise_model is not None:
        # Get singles samplers
        if hasattr(dataset.noise_model, 'singles'):
            noise_samplers.update(dataset.noise_model.singles)

    # Create surrogate
    surrogate = SCMProcessSurrogate(
        node_specs=node_specs,
        input_labels=process_config['input_labels'],
        output_labels=process_config['output_labels'],
        noise_samplers=noise_samplers,
        device=device
    )

    # Freeze parameters (even though there are none, for consistency)
    for param in surrogate.parameters():
        param.requires_grad = False

    return surrogate


if __name__ == '__main__':
    print("Testing SCMProcessSurrogate...")

    # Test with laser process
    from controller_optimization.configs.processes_config import PROCESSES

    laser_config = PROCESSES[0]  # Assuming laser is first

    print(f"\nCreating SCM surrogate for process: {laser_config['name']}")
    surrogate = create_scm_surrogate_for_process(laser_config, device='cpu')

    print(f"  Input labels: {surrogate.input_labels}")
    print(f"  Output labels: {surrogate.output_labels}")
    print(f"  Topological order: {surrogate.order}")

    # Test forward pass
    batch_size = 4
    input_dim = len(laser_config['input_labels'])

    # Create dummy inputs
    x = torch.randn(batch_size, input_dim)
    print(f"\nTest forward pass with batch_size={batch_size}")
    print(f"  Input shape: {x.shape}")

    mean, variance = surrogate(x)
    print(f"  Output mean shape: {mean.shape}")
    print(f"  Output variance shape: {variance.shape}")
    print(f"  Mean values: {mean}")
    print(f"  Variance values: {variance}")

    # Test differentiability
    x_grad = torch.randn(batch_size, input_dim, requires_grad=True)
    mean_grad, variance_grad = surrogate(x_grad)
    loss = mean_grad.sum()
    loss.backward()

    print(f"\nDifferentiability test:")
    print(f"  Gradient exists: {x_grad.grad is not None}")
    if x_grad.grad is not None:
        print(f"  Gradient shape: {x_grad.grad.shape}")
        print(f"  Gradient norm: {x_grad.grad.norm().item():.6f}")

    print("\n✓ SCMProcessSurrogate test passed!")
