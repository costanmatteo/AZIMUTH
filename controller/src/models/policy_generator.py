"""
Policy Generator: genera parametri di input ottimali per il processo successivo.

Architettura simile a UncertaintyPredictor ma con output bounded.
Output è limitato tra min e max derivati dall'UncertaintyPredictor.
"""

import torch
import torch.nn as nn


class PolicyGenerator(nn.Module):
    """
    Policy generator per controller optimization with bounded outputs.

    The output layer uses tanh activation to produce values in [-1, 1],
    which are then scaled to [output_min, output_max] bounds derived from
    the UncertaintyPredictor's training data.

    Args:
        input_size (int): Dimensione input concatenato
        hidden_sizes (list): Hidden layers
        output_size (int): Dimensione output (input per processo successivo)
        dropout_rate (float): Dropout
        use_batchnorm (bool): Batch normalization
        output_min (torch.Tensor): Minimum bounds for each output dimension
        output_max (torch.Tensor): Maximum bounds for each output dimension
        debug (bool): Enable debug output
        name (str): Name for identifying which policy generator
    """

    def __init__(self, input_size, hidden_sizes, output_size,
                 dropout_rate=0.1, use_batchnorm=False,
                 output_min=None, output_max=None,
                 debug: bool = False,
                 name: str = ""):
        super(PolicyGenerator, self).__init__()

        self.output_size = output_size
        self.debug = debug
        self.name = name

        # Register bounds as buffers (not parameters, but saved with model)
        if output_min is not None:
            self.register_buffer('output_min', output_min)
        else:
            self.register_buffer('output_min', None)

        if output_max is not None:
            self.register_buffer('output_max', output_max)
        else:
            self.register_buffer('output_max', None)

        # Input normalization to prevent tanh saturation from scale mismatch
        # (prev_outputs_mean, prev_outputs_var, and env inputs can differ by orders of magnitude)
        self.input_norm = nn.LayerNorm(input_size)

        # Build shared hidden layers (same architecture as UncertaintyPredictor)
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.SiLU())  # Same activation as UncertaintyPredictor
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        self.network = nn.Sequential(*layers)

        # Single output head (no variance head)
        self.output_head = nn.Linear(prev_size, output_size)

    def set_bounds(self, output_min, output_max):
        """
        Set or update output bounds.

        Args:
            output_min (torch.Tensor): Minimum bounds, shape (output_size,)
            output_max (torch.Tensor): Maximum bounds, shape (output_size,)
        """
        device = next(self.parameters()).device
        self.output_min = output_min.to(device)
        self.output_max = output_max.to(device)

    def _apply_bounds(self, raw_actions):
        """
        Apply tanh activation and scale to output bounds.

        Args:
            raw_actions (torch.Tensor): Raw output from the network

        Returns:
            tuple: (actions, tanh_out, normalized)
        """
        tanh_out = torch.tanh(raw_actions)
        normalized = 0.5 * (tanh_out + 1.0)
        actions = self.output_min + normalized * (self.output_max - self.output_min)
        return actions, tanh_out, normalized

    def _debug_forward(self, x, features, raw_actions, tanh_out, normalized, actions):
        """Print debug information for forward pass."""
        with torch.no_grad():
            if tanh_out is not None:
                print(f"\n{'='*60}")
                print(f"PolicyGenerator DEBUG [{self.name}]")
                print(f"{'='*60}")
                print(f"Input x: mean={x.mean().item():.4f}, std={x.std().item():.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}")
                print(f"Features: mean={features.mean().item():.4f}, std={features.std().item():.4f}")
                print(f"Raw actions (pre-tanh): mean={raw_actions.mean().item():.4f}, std={raw_actions.std().item():.4f}, min={raw_actions.min().item():.4f}, max={raw_actions.max().item():.4f}")
                print(f"Tanh output: mean={tanh_out.mean().item():.4f}, std={tanh_out.std().item():.4f}, min={tanh_out.min().item():.4f}, max={tanh_out.max().item():.4f}")
                print(f"Normalized [0,1]: mean={normalized.mean().item():.4f}, std={normalized.std().item():.4f}, min={normalized.min().item():.4f}, max={normalized.max().item():.4f}")
                print(f"Bounds: min={self.output_min.cpu().numpy()}, max={self.output_max.cpu().numpy()}")
                print(f"Final actions: mean={actions.mean().item():.4f}, std={actions.std().item():.4f}")
                for i in range(actions.shape[1]):
                    print(f"  Dim {i}: mean={actions[:, i].mean().item():.4f}, min={actions[:, i].min().item():.4f}, max={actions[:, i].max().item():.4f}")
                # Check for saturation
                saturated_low = (tanh_out < -0.99).float().mean().item() * 100
                saturated_high = (tanh_out > 0.99).float().mean().item() * 100
                print(f"Tanh saturation: {saturated_low:.1f}% at -1, {saturated_high:.1f}% at +1")
                print(f"{'='*60}\n")
            else:
                print(f"\n[PolicyGenerator DEBUG - {self.name}] UNBOUNDED MODE")
                print(f"  Raw actions: mean={raw_actions.mean().item():.4f}, std={raw_actions.std().item():.4f}")

    def forward(self, x):
        """
        Forward pass with bounded output.

        Uses tanh activation followed by affine scaling to enforce bounds:
            normalized = 0.5 * (tanh(raw) + 1)  -> [0, 1]
            bounded = min + normalized * (max - min)  -> [min, max]

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)

        Returns:
            torch.Tensor: Bounded actions for next process, shape (batch_size, output_size)
        """
        x = self.input_norm(x)
        features = self.network(x)
        raw_actions = self.output_head(features)

        # Apply bounded activation if bounds are set
        if self.output_min is not None and self.output_max is not None:
            actions, tanh_out, normalized = self._apply_bounds(raw_actions)

            if self.debug:
                self._debug_forward(x, features, raw_actions, tanh_out, normalized, actions)
        else:
            # Fallback to unbounded (for backward compatibility)
            actions = raw_actions
            if self.debug:
                self._debug_forward(x, features, raw_actions, None, None, actions)

        return actions

    def debug_gradients(self):
        """Print gradient statistics for debugging."""
        print(f"\n{'='*60}")
        print(f"GRADIENT DEBUG [{self.name}]")
        print(f"{'='*60}")

        total_grad_norm = 0.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm ** 2
                print(f"  {name}: grad_norm={grad_norm:.6f}, "
                      f"grad_mean={param.grad.mean().item():.6f}, "
                      f"grad_std={param.grad.std().item():.6f}")
            else:
                print(f"  {name}: NO GRADIENT")

        total_grad_norm = total_grad_norm ** 0.5
        print(f"Total gradient norm: {total_grad_norm:.6f}")
        print(f"{'='*60}\n")

    def debug_weights(self):
        """Print weight statistics for debugging."""
        print(f"\n{'='*60}")
        print(f"WEIGHT DEBUG [{self.name}]")
        print(f"{'='*60}")

        for name, param in self.named_parameters():
            print(f"  {name}: mean={param.data.mean().item():.6f}, "
                  f"std={param.data.std().item():.6f}, "
                  f"min={param.data.min().item():.6f}, "
                  f"max={param.data.max().item():.6f}")
        print(f"{'='*60}\n")
