"""
Quick test script for process-based configuration system

This script verifies that the process configuration system works correctly
without requiring actual CSV files.

Run: python test_process_config.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_process_config():
    """Test process configuration loading and validation"""
    print("="*70)
    print("Testing Process Configuration System")
    print("="*70)

    from configs.process_config import (
        get_process_config,
        get_available_processes,
        set_output_columns,
        get_column_mapping
    )

    # Test 1: Get available processes
    print("\n[TEST 1] Available processes:")
    processes = get_available_processes()
    print(f"  ✓ Found {len(processes)} processes: {', '.join(processes)}")
    assert len(processes) == 5, "Should have 5 processes"

    # Test 2: Get configuration for each process
    print("\n[TEST 2] Process configurations:")
    for process in processes:
        config = get_process_config(process)
        print(f"  ✓ {process:12s} -> {config['filename']:20s} (prefix: {config['prefix']})")
        assert 'filename' in config
        assert 'metadata_columns' in config
        assert 'output_columns' in config

    # Test 3: Set output columns
    print("\n[TEST 3] Setting output columns:")
    set_output_columns('laser', ['Temperature', 'Quality'])
    config = get_process_config('laser')
    assert config['output_columns'] == ['Temperature', 'Quality']
    print(f"  ✓ Laser output columns: {config['output_columns']}")

    # Test 4: Column mapping
    print("\n[TEST 4] Automatic column mapping:")

    # Simulate CSV columns
    csv_columns = [
        'WA', 'PanelNr', 'TimeStamp',  # Metadata
        'Process_1', 'Machine',         # Metadata
        'Temperature', 'Quality',       # Outputs (as set above)
        'Power', 'Speed', 'Pressure'    # Should become inputs
    ]

    input_cols, output_cols, metadata_cols = get_column_mapping('laser', csv_columns)

    print(f"  Metadata (excluded): {metadata_cols}")
    print(f"  Outputs (targets):   {output_cols}")
    print(f"  Inputs (features):   {input_cols}")

    assert 'Power' in input_cols, "Power should be input"
    assert 'Speed' in input_cols, "Speed should be input"
    assert 'Pressure' in input_cols, "Pressure should be input"
    assert 'Temperature' in output_cols, "Temperature should be output"
    assert 'Quality' in output_cols, "Quality should be output"
    assert 'WA' in metadata_cols, "WA should be metadata"
    assert 'WA' not in input_cols, "WA should not be input"
    print("  ✓ Column mapping correct!")

    # Test 5: Invalid process name
    print("\n[TEST 5] Error handling:")
    try:
        get_process_config('invalid_process')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Correctly raised error for invalid process")
        print(f"    Message: {str(e)[:60]}...")

    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)


def test_data_loading_imports():
    """Test that data loading functions can be imported"""
    print("\n[TEST 6] Import test:")

    try:
        from src.data import load_process_data, DataPreprocessor, load_csv_data

        print("  ✓ load_process_data imported")
        print("  ✓ DataPreprocessor imported")
        print("  ✓ load_csv_data imported")
    except ModuleNotFoundError as e:
        if 'torch' in str(e):
            print("  ⚠ PyTorch not installed (expected in some environments)")
            print("  ✓ But configuration system works independently!")
        else:
            raise


if __name__ == '__main__':
    try:
        test_process_config()
        test_data_loading_imports()

        print("\n" + "="*70)
        print("Process-based configuration system is working correctly!")
        print("="*70)
        print("\nNext steps:")
        print("1. Place your CSV files in 'src/data/raw/'")
        print("2. Edit 'configs/process_config.py' to set output_columns")
        print("3. Run 'python examples/train_with_process.py'")
        print("\nFor more info, see: docs/PROCESS_BASED_TRAINING.md")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
