"""Debug script to check if policy generator is producing non-zero outputs."""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from controller_optimization.configs.controller_config import CONTROLLER_CONFIG
from controller_optimization.src.utils.process_chain import ProcessChain
from controller_optimization.src.utils.surrogate import StructuralReliabilitySurrogate

# Load configuration
device = 'cpu'

# Initialize process chain
process_chain = ProcessChain(
    processes_config=CONTROLLER_CONFIG['processes'],
    target_trajectory_path=CONTROLLER_CONFIG['training']['target_trajectory_path'],
    scenario_encoder_config=None,
    device=device
)

# Forward pass with scenario 0
print("Testing policy generator outputs...")
print("=" * 70)

with torch.no_grad():
    trajectory = process_chain.forward(batch_size=1, scenario_idx=0)

    for process_name, data in trajectory.items():
        print(f"\n{process_name.upper()}:")
        print(f"  Inputs shape: {data['inputs'].shape}")
        print(f"  Inputs values: {data['inputs']}")
        print(f"  Inputs mean: {data['inputs'].mean().item():.6f}")
        print(f"  Inputs std: {data['inputs'].std().item():.6f}")
        print(f"  Inputs min: {data['inputs'].min().item():.6f}")
        print(f"  Inputs max: {data['inputs'].max().item():.6f}")

print("\n" + "=" * 70)
print("\nChecking policy generator parameters...")
for i, pg in enumerate(process_chain.policy_generators):
    print(f"\nPolicy Generator {i}:")
    has_nonzero = False
    for name, param in pg.named_parameters():
        if param.abs().sum() > 0:
            has_nonzero = True
            print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    if not has_nonzero:
        print("  WARNING: All parameters are zero!")
