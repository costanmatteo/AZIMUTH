"""
Process-specific configuration for Uncertainty Quantification

This module defines the configuration for each manufacturing process,
including metadata labels, file paths, and automatic input/output column mapping.

For each process:
- Metadata columns (like WA, Panel, Timestamp) are excluded from training
- Output columns are specified explicitly
- All remaining columns are automatically used as inputs
"""

PROCESS_CONFIGS = {
    "laser": {
        # Process identification
        "process_label": "Laser",
        "hidden_label": "Process_1",
        "machine_label": "Machine",
        "WA_label": "WA",
        "panel_label": "PanelNr",
        "PaPos_label": "PaPosNr",
        "date_label": ["TimeStamp", "CreateDate 1"],
        "date_format": "%m/%d/%y %I:%M %p",

        # File configuration
        "prefix": "las",
        "filename": "laser.csv",
        "sep": ",",
        "header": 0,

        # Column mapping for ML
        "metadata_columns": [
            # Columns to exclude from both input and output (identifiers, timestamps)
            "WA", "PanelNr", "PaPosNr", "TimeStamp", "CreateDate 1",
            "Process_1", "Machine"
        ],
        "output_columns": [
            # Target variables to predict
            # TODO: Define based on your specific use case
            # Example: ["Temperature", "Quality_Score"]
        ],
        # input_columns are automatically determined:
        # input_columns = all_columns - metadata_columns - output_columns
    },

    "plasma": {
        "process_label": "Plasma",
        "hidden_label": "Process_2",
        "machine_label": "Machine",
        "WA_label": "WA",
        "panel_label": "PanelNummer",
        "PaPos_label": "Position",
        "date_label": ["Buchungsdatum"],
        "date_format": "%m/%d/%y %I:%M %p",

        "prefix": "pla",
        "filename": "plasma_fixed.csv",
        "sep": ",",
        "header": 0,

        "metadata_columns": [
            "WA", "PanelNummer", "Position", "Buchungsdatum",
            "Process_2", "Machine"
        ],
        "output_columns": [
            # TODO: Define target variables
        ],
    },

    "galvanic": {
        "process_label": "Galvanic",
        "hidden_label": "Process_3",
        "machine_label": None,
        "WA_label": "WA",
        "panel_label": "Panelnr",
        "PaPos_label": "PaPosNr",
        "date_label": ["Date/Time Stamp"],
        "date_format": "%m/%d/%y %I:%M %p",

        "prefix": "gal",
        "filename": "galvanik.csv",
        "sep": ",",
        "header": 0,

        "metadata_columns": [
            "WA", "Panelnr", "PaPosNr", "Date/Time Stamp", "Process_3"
        ],
        "output_columns": [
            # TODO: Define target variables
        ],
    },

    "multibond": {
        "process_label": "Multibond",
        "hidden_label": "Process_4",
        "machine_label": None,
        "WA_label": "WA",
        "panel_label": None,
        "PaPos_label": "PaPosNr",
        "date_label": ["t_StartDateTime"],
        "date_format": "%m/%d/%y %I:%M %p",

        "prefix": "mul",
        "filename": "multibond.csv",
        "sep": ",",
        "header": 0,

        "metadata_columns": [
            "WA", "PaPosNr", "t_StartDateTime", "Process_4"
        ],
        "output_columns": [
            # TODO: Define target variables
        ],
    },

    "microetch": {
        "process_label": "Microetch",
        "hidden_label": "Process_5",
        "machine_label": None,
        "WA_label": "WA",
        "panel_label": None,
        "PaPos_label": "PaPosNr",
        "date_label": ["CreateDate"],
        "date_format": "%d.%m.%Y %H:%M:%S",

        "prefix": "mic",
        "filename": "microetch.csv",
        "sep": ",",
        "header": 0,

        "metadata_columns": [
            "WA", "PaPosNr", "CreateDate", "Process_5"
        ],
        "output_columns": [
            # TODO: Define target variables
        ],
    },
}


def get_process_config(process_name):
    """
    Get configuration for a specific process.

    Args:
        process_name (str): Name of the process ('laser', 'plasma', etc.)

    Returns:
        dict: Process configuration

    Raises:
        ValueError: If process_name is not recognized
    """
    if process_name not in PROCESS_CONFIGS:
        available = ', '.join(PROCESS_CONFIGS.keys())
        raise ValueError(
            f"Unknown process '{process_name}'. "
            f"Available processes: {available}"
        )

    return PROCESS_CONFIGS[process_name]


def get_available_processes():
    """
    Get list of available process names.

    Returns:
        list: List of process names
    """
    return list(PROCESS_CONFIGS.keys())


def set_output_columns(process_name, output_columns):
    """
    Set output columns for a specific process.

    Args:
        process_name (str): Name of the process
        output_columns (list): List of column names to use as outputs
    """
    if process_name not in PROCESS_CONFIGS:
        raise ValueError(f"Unknown process '{process_name}'")

    PROCESS_CONFIGS[process_name]['output_columns'] = output_columns
    print(f"Output columns for '{process_name}' set to: {output_columns}")


def get_column_mapping(process_name, csv_columns):
    """
    Automatically determine input and output columns for a process.

    Args:
        process_name (str): Name of the process
        csv_columns (list): List of all columns in the CSV file

    Returns:
        tuple: (input_columns, output_columns, metadata_columns)
    """
    config = get_process_config(process_name)

    metadata_columns = config.get('metadata_columns', [])
    output_columns = config.get('output_columns', [])

    # Filter to only include columns that actually exist in the CSV
    metadata_columns = [col for col in metadata_columns if col in csv_columns]
    output_columns = [col for col in output_columns if col in csv_columns]

    # Input columns = all columns - metadata - output
    excluded = set(metadata_columns + output_columns)
    input_columns = [col for col in csv_columns if col not in excluded]

    return input_columns, output_columns, metadata_columns
