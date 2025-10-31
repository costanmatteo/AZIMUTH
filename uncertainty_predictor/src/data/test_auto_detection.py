"""
Test script for auto-detection feature.

Tests that the system can automatically detect process type from Excel files.
"""

import os
from excel_to_csv_converter import auto_detect_process, convert_excel_auto
from preprocessing import load_data

# Create test directory
TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_data')


def test_auto_detect():
    """Test auto-detection with existing test files"""
    print("="*60)
    print("Testing Auto-Detection Feature")
    print("="*60 + "\n")

    # Check if test files exist
    laser_file = os.path.join(TEST_DIR, 'sample_laser.xlsx')
    plasma_file = os.path.join(TEST_DIR, 'sample_plasma.xlsx')

    if not os.path.exists(laser_file):
        print(f"⚠️  Test file not found: {laser_file}")
        print("Run test_converter.py first to create test files.")
        return

    # Test 1: Auto-detect laser
    print("Test 1: Auto-detecting laser file...")
    try:
        detected_process = auto_detect_process(laser_file)
        if detected_process == 'laser':
            print(f"✓ Correctly detected: {detected_process}")
        else:
            print(f"✗ FAILED: Expected 'laser', got '{detected_process}'")
    except Exception as e:
        print(f"✗ ERROR: {e}")

    print("\n" + "-"*60 + "\n")

    # Test 2: Auto-detect plasma
    if os.path.exists(plasma_file):
        print("Test 2: Auto-detecting plasma file...")
        try:
            detected_process = auto_detect_process(plasma_file)
            if detected_process == 'plasma':
                print(f"✓ Correctly detected: {detected_process}")
            else:
                print(f"✗ FAILED: Expected 'plasma', got '{detected_process}'")
        except Exception as e:
            print(f"✗ ERROR: {e}")

        print("\n" + "-"*60 + "\n")

    # Test 3: Full conversion with auto-detect
    print("Test 3: Full conversion with auto-detection...")
    try:
        csv_path = convert_excel_auto(laser_file, TEST_DIR)
        print(f"✓ Conversion successful!")
        print(f"  Output: {csv_path}")

        # Verify the CSV was created
        if os.path.exists(csv_path):
            print(f"✓ CSV file exists")
        else:
            print(f"✗ CSV file not created")

    except Exception as e:
        print(f"✗ ERROR: {e}")

    print("\n" + "-"*60 + "\n")

    # Test 4: Load data using new load_data function
    print("Test 4: Testing load_data() with Excel file...")
    try:
        X, y = load_data(
            laser_file,
            input_columns=['las_machine', 'las_wa', 'las_panel'],
            output_columns=['las_papos'],
            output_dir=TEST_DIR
        )
        print(f"✓ Data loaded successfully!")
        print(f"  Samples: {len(X)}")
        print(f"  Input features: {X.shape[1]}")
        print(f"  Output features: {y.shape[1]}")
    except Exception as e:
        print(f"✗ ERROR: {e}")

    print("\n" + "="*60)
    print("Auto-Detection Tests Complete!")
    print("="*60)


if __name__ == "__main__":
    test_auto_detect()
