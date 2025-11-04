"""
Simple test script to verify the Excel to CSV converter is working.

This creates a sample Excel file and converts it.
"""

import pandas as pd
import os
from excel_to_csv_converter import convert_process_file, PROCESS_CONFIGS

# Create test directory
TEST_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
os.makedirs(TEST_DIR, exist_ok=True)


def create_sample_laser_data():
    """Create a sample laser Excel file for testing"""
    data = {
        'TimeStamp': ['10/31/24 2:30 PM', '10/31/24 3:45 PM', '10/31/24 4:15 PM'],
        'Machine': ['Machine_A', 'Machine_B', 'Machine_A'],
        'WA': ['WA001', 'WA002', 'WA003'],
        'PanelNr': ['P001', 'P002', 'P003'],
        'PaPosNr': [1, 2, 3],
        'Laser': ['Laser_1', 'Laser_1', 'Laser_2'],
        'Process_1': ['Proc_A', 'Proc_A', 'Proc_B'],
        'ExtraColumn': ['ignore', 'this', 'column']  # This should be ignored
    }

    df = pd.DataFrame(data)
    output_path = os.path.join(TEST_DIR, 'sample_laser.xlsx')

    df.to_excel(output_path, index=False)
    print(f"Created sample laser file: {output_path}")

    return output_path


def create_sample_plasma_data():
    """Create a sample plasma Excel file for testing"""
    data = {
        'Buchungsdatum': ['10/31/24 2:30 PM', '10/31/24 3:45 PM'],
        'Machine': ['Machine_C', 'Machine_D'],
        'WA': ['WA004', 'WA005'],
        'PanelNummer': ['P004', 'P005'],
        'Position': [1, 2],
        'Plasma': ['Plasma_1', 'Plasma_2'],
        'Process_2': ['Proc_C', 'Proc_D'],
    }

    df = pd.DataFrame(data)
    output_path = os.path.join(TEST_DIR, 'sample_plasma.xlsx')

    df.to_excel(output_path, index=False)
    print(f"Created sample plasma file: {output_path}")

    return output_path


def test_conversion():
    """Test the conversion process"""
    print("="*60)
    print("Testing Excel to CSV Converter")
    print("="*60 + "\n")

    # Create sample data files
    print("Step 1: Creating sample Excel files...")
    laser_file = create_sample_laser_data()
    plasma_file = create_sample_plasma_data()

    print("\nStep 2: Converting files...\n")

    # Convert laser file
    try:
        output_laser = convert_process_file(
            process_name='laser',
            input_path=laser_file,
            output_dir=TEST_DIR
        )
        print(f"\n✓ Laser conversion successful!")
        print(f"  Output: {output_laser}")

        # Show the result
        df_laser = pd.read_csv(output_laser)
        print(f"  Rows: {len(df_laser)}")
        print(f"  Columns: {list(df_laser.columns)}")
        print("\n  First row:")
        print(df_laser.head(1).to_string(index=False))

    except Exception as e:
        print(f"\n✗ Laser conversion failed: {e}")

    print("\n" + "-"*60 + "\n")

    # Convert plasma file
    try:
        output_plasma = convert_process_file(
            process_name='plasma',
            input_path=plasma_file,
            output_dir=TEST_DIR
        )
        print(f"\n✓ Plasma conversion successful!")
        print(f"  Output: {output_plasma}")

        # Show the result
        df_plasma = pd.read_csv(output_plasma)
        print(f"  Rows: {len(df_plasma)}")
        print(f"  Columns: {list(df_plasma.columns)}")
        print("\n  First row:")
        print(df_plasma.head(1).to_string(index=False))

    except Exception as e:
        print(f"\n✗ Plasma conversion failed: {e}")

    print("\n" + "="*60)
    print("Test Complete!")
    print(f"Test files saved in: {TEST_DIR}")
    print("="*60)


if __name__ == "__main__":
    test_conversion()
