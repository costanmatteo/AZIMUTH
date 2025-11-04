"""
Example script showing how to use the Excel to CSV converter.

Update the file paths below and run this script to convert your Excel files.
"""

from excel_to_csv_converter import (
    convert_process_file,
    convert_all_processes,
    PROCESS_CONFIGS
)
import os

# Get the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(CURRENT_DIR, 'raw')
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'processed')


def example_single_file():
    """
    Example: Convert a single process file
    """
    print("Example 1: Converting a single file\n")

    # Convert laser data
    input_file = '/path/to/your/laser.xlsx'  # UPDATE THIS PATH
    output_dir = OUTPUT_DIR

    try:
        output_path = convert_process_file(
            process_name='laser',
            input_path=input_file,
            output_dir=output_dir
        )
        print(f"\nSuccess! Output saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")


def example_multiple_files():
    """
    Example: Convert multiple process files at once
    """
    print("Example 2: Converting multiple files\n")

    # Define your input files
    # UPDATE THESE PATHS with your actual Excel file locations
    input_files = {
        'laser': '/path/to/your/laser.xlsx',
        'plasma': '/path/to/your/plasma.xlsx',
        'galvanic': '/path/to/your/galvanic.xlsx',
        'multibond': '/path/to/your/multibond.xlsx',
        'microetch': '/path/to/your/microetch.xlsx',
    }

    # Convert all files
    results = convert_all_processes(input_files, OUTPUT_DIR)

    # Print results
    print("\n" + "="*60)
    print("CONVERSION RESULTS")
    print("="*60)

    for process, output_path in results.items():
        if output_path:
            print(f"✓ {process:15} -> {output_path}")
        else:
            print(f"✗ {process:15} -> FAILED")


def example_check_columns():
    """
    Example: Check what columns will be extracted for each process
    """
    print("Example 3: Checking expected columns\n")

    for process_name, config in PROCESS_CONFIGS.items():
        print(f"\n{process_name.upper()}:")
        print(f"  Input Excel columns needed:")

        if config.get('process_label'):
            print(f"    - {config['process_label']} (process)")
        if config.get('hidden_label'):
            print(f"    - {config['hidden_label']} (hidden process)")
        if config.get('machine_label'):
            print(f"    - {config['machine_label']} (machine)")
        if config.get('WA_label'):
            print(f"    - {config['WA_label']} (work area)")
        if config.get('panel_label'):
            print(f"    - {config['panel_label']} (panel)")
        if config.get('PaPos_label'):
            print(f"    - {config['PaPos_label']} (position)")

        date_labels = config.get('date_label', [])
        if date_labels:
            print(f"    - One of: {', '.join(date_labels)} (timestamp)")

        print(f"  Output CSV: {config['filename']}")
        print(f"  Date format: {config['date_format']}")


def example_custom_output_location():
    """
    Example: Convert with custom output location
    """
    print("Example 4: Custom output location\n")

    input_file = '/path/to/your/plasma.xlsx'  # UPDATE THIS
    custom_output = '/path/to/custom/output/directory'  # UPDATE THIS

    try:
        output_path = convert_process_file(
            process_name='plasma',
            input_path=input_file,
            output_dir=custom_output
        )
        print(f"Success! Output saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Excel to CSV Converter - Usage Examples")
    print("="*60 + "\n")

    # Show available process configurations
    print("Available processes:")
    for process_name in PROCESS_CONFIGS.keys():
        print(f"  - {process_name}")

    print("\n" + "="*60 + "\n")

    # Run example 3 to show expected columns
    example_check_columns()

    print("\n" + "="*60)
    print("\nTo convert your files:")
    print("1. Update the file paths in this script")
    print("2. Uncomment one of the example functions below")
    print("3. Run: python convert_example.py")
    print("="*60)

    # Uncomment the example you want to run:
    # example_single_file()
    # example_multiple_files()
    # example_custom_output_location()
