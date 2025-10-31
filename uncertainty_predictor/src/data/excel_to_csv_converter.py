"""
Excel to CSV Converter for Manufacturing Process Data

This script converts Excel files containing manufacturing process data
into structured CSV files suitable for neural network training.

Supports multiple process types:
- Laser
- Plasma
- Galvanic
- Multibond
- Microetch
"""

import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Process configurations
PROCESS_CONFIGS = {
    "laser": {
        "process_label": "Laser",
        "hidden_label": "Process_1",
        "machine_label": "Machine",
        "WA_label": "WA",
        "panel_label": "PanelNr",
        "PaPos_label": "PaPosNr",
        "date_label": ["TimeStamp", "CreateDate 1"],
        "date_format": "%m/%d/%y %I:%M %p",
        "prefix": "las",
        "filename": "laser.csv",
        "sep": ",",
        "header": 0
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
        "header": 0
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
        "header": 0
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
        "header": 0
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
        "header": 0
    }
}


def auto_detect_process(input_path: str, header: int = 0) -> str:
    """
    Automatically detect which process configuration to use based on Excel columns.

    Tries all process configurations and returns the one with the most matching columns.

    Args:
        input_path (str): Path to Excel file
        header (int): Header row number (default: 0)

    Returns:
        str: Best matching process name ('laser', 'plasma', etc.)

    Raises:
        ValueError: If no suitable configuration is found
    """
    logger.info(f"Auto-detecting process type for: {input_path}")

    # Load Excel to get column names
    try:
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path, nrows=0, header=header)
        else:
            df = pd.read_excel(input_path, nrows=0, header=header)

        available_columns = set(df.columns)
        logger.info(f"Available columns in file: {list(available_columns)}")

    except Exception as e:
        raise ValueError(f"Cannot read file {input_path}: {e}")

    # Score each configuration
    scores = {}

    for process_name, config in PROCESS_CONFIGS.items():
        score = 0
        required_columns = []

        # Check each label type
        if config.get('process_label'):
            required_columns.append(config['process_label'])
            if config['process_label'] in available_columns:
                score += 2  # Process label is important

        if config.get('hidden_label'):
            required_columns.append(config['hidden_label'])
            if config['hidden_label'] in available_columns:
                score += 2  # Hidden label is important

        if config.get('machine_label'):
            required_columns.append(config['machine_label'])
            if config['machine_label'] in available_columns:
                score += 1

        if config.get('WA_label'):
            required_columns.append(config['WA_label'])
            if config['WA_label'] in available_columns:
                score += 2  # WA is critical

        if config.get('panel_label'):
            required_columns.append(config['panel_label'])
            if config['panel_label'] in available_columns:
                score += 1

        if config.get('PaPos_label'):
            required_columns.append(config['PaPos_label'])
            if config['PaPos_label'] in available_columns:
                score += 1

        # Check date labels (at least one must match)
        date_labels = config.get('date_label', [])
        date_match = any(dl in available_columns for dl in date_labels)
        if date_match:
            score += 1

        scores[process_name] = {
            'score': score,
            'required_columns': required_columns,
            'matching_columns': [col for col in required_columns if col in available_columns]
        }

        logger.info(f"  {process_name}: score={score}, matching={len(scores[process_name]['matching_columns'])}/{len(required_columns)}")

    # Find best match
    best_process = max(scores.items(), key=lambda x: x[1]['score'])
    best_name = best_process[0]
    best_score = best_process[1]['score']

    if best_score == 0:
        raise ValueError(
            f"No matching process configuration found for {input_path}. "
            f"Available columns: {list(available_columns)}"
        )

    logger.info(f"✓ Best match: '{best_name}' (score: {best_score})")
    logger.info(f"  Matching columns: {scores[best_name]['matching_columns']}")

    return best_name


class ExcelToCSVConverter:
    """
    Converter for transforming Excel manufacturing data to structured CSV files.

    Args:
        config (dict): Process configuration dictionary
        input_path (str): Path to input Excel file
        output_dir (str): Directory for output CSV file
    """

    def __init__(self, config: Dict, input_path: str, output_dir: str = "./output"):
        self.config = config
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def load_excel(self) -> pd.DataFrame:
        """
        Load Excel file into pandas DataFrame.

        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Loading Excel file: {self.input_path}")

        try:
            # Try reading as Excel file
            if self.input_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(
                    self.input_path,
                    header=self.config.get('header', 0)
                )
            # If it's already a CSV, read it as CSV
            elif self.input_path.endswith('.csv'):
                self.df = pd.read_csv(
                    self.input_path,
                    sep=self.config.get('sep', ','),
                    header=self.config.get('header', 0)
                )
            else:
                raise ValueError(f"Unsupported file format: {self.input_path}")

            logger.info(f"Successfully loaded {len(self.df)} rows and {len(self.df.columns)} columns")
            logger.info(f"Columns: {list(self.df.columns)}")
            return self.df

        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise

    def extract_relevant_columns(self) -> pd.DataFrame:
        """
        Extract only the relevant columns based on configuration.

        Returns:
            pd.DataFrame: DataFrame with only relevant columns
        """
        logger.info("Extracting relevant columns...")

        relevant_cols = []
        column_mapping = {}

        # Process label
        if self.config.get('process_label'):
            relevant_cols.append(self.config['process_label'])
            column_mapping[self.config['process_label']] = f"{self.config['prefix']}_process"

        # Hidden label
        if self.config.get('hidden_label'):
            relevant_cols.append(self.config['hidden_label'])
            column_mapping[self.config['hidden_label']] = f"{self.config['prefix']}_hidden_process"

        # Machine label
        if self.config.get('machine_label'):
            relevant_cols.append(self.config['machine_label'])
            column_mapping[self.config['machine_label']] = f"{self.config['prefix']}_machine"

        # WA label
        if self.config.get('WA_label'):
            relevant_cols.append(self.config['WA_label'])
            column_mapping[self.config['WA_label']] = f"{self.config['prefix']}_wa"

        # Panel label
        if self.config.get('panel_label'):
            relevant_cols.append(self.config['panel_label'])
            column_mapping[self.config['panel_label']] = f"{self.config['prefix']}_panel"

        # PaPos label
        if self.config.get('PaPos_label'):
            relevant_cols.append(self.config['PaPos_label'])
            column_mapping[self.config['PaPos_label']] = f"{self.config['prefix']}_papos"

        # Date labels - try each one until we find one that exists
        date_labels = self.config.get('date_label', [])
        date_col_found = None
        for date_label in date_labels:
            if date_label in self.df.columns:
                relevant_cols.append(date_label)
                column_mapping[date_label] = f"{self.config['prefix']}_timestamp"
                date_col_found = date_label
                break

        if not date_col_found and date_labels:
            logger.warning(f"None of the date columns {date_labels} found in the data")

        # Filter only existing columns
        existing_cols = [col for col in relevant_cols if col in self.df.columns]
        missing_cols = [col for col in relevant_cols if col not in self.df.columns]

        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")

        if not existing_cols:
            raise ValueError("No relevant columns found in the Excel file!")

        logger.info(f"Extracting columns: {existing_cols}")

        # Extract and rename columns
        df_extracted = self.df[existing_cols].copy()
        df_extracted = df_extracted.rename(columns=column_mapping)

        return df_extracted

    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse date columns according to the specified format.

        Args:
            df (pd.DataFrame): DataFrame with date column

        Returns:
            pd.DataFrame: DataFrame with parsed dates
        """
        date_format = self.config.get('date_format')
        timestamp_col = f"{self.config['prefix']}_timestamp"

        if timestamp_col in df.columns and date_format:
            logger.info(f"Parsing dates with format: {date_format}")
            try:
                df[timestamp_col] = pd.to_datetime(
                    df[timestamp_col],
                    format=date_format,
                    errors='coerce'
                )

                # Count parsing failures
                null_count = df[timestamp_col].isna().sum()
                if null_count > 0:
                    logger.warning(f"{null_count} dates could not be parsed")

            except Exception as e:
                logger.warning(f"Error parsing dates: {e}. Trying automatic detection...")
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by removing rows with critical missing values.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("Cleaning data...")

        initial_rows = len(df)

        # Remove completely empty rows
        df = df.dropna(how='all')

        # Optionally remove rows with missing WA (critical identifier)
        wa_col = f"{self.config['prefix']}_wa"
        if wa_col in df.columns:
            df = df.dropna(subset=[wa_col])

        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} rows during cleaning")

        return df

    def save_csv(self, df: pd.DataFrame) -> str:
        """
        Save the processed DataFrame to CSV.

        Args:
            df (pd.DataFrame): Processed DataFrame

        Returns:
            str: Output file path
        """
        output_path = os.path.join(self.output_dir, self.config['filename'])

        logger.info(f"Saving to: {output_path}")

        df.to_csv(
            output_path,
            sep=self.config.get('sep', ','),
            index=False
        )

        logger.info(f"Successfully saved {len(df)} rows to {output_path}")

        return output_path

    def convert(self) -> str:
        """
        Execute the full conversion pipeline.

        Returns:
            str: Path to output CSV file
        """
        logger.info(f"Starting conversion for process: {self.config['process_label']}")

        # Load Excel
        self.load_excel()

        # Extract relevant columns
        df_extracted = self.extract_relevant_columns()

        # Parse dates
        df_parsed = self.parse_dates(df_extracted)

        # Clean data
        df_cleaned = self.clean_data(df_parsed)

        # Save to CSV
        output_path = self.save_csv(df_cleaned)

        logger.info("Conversion completed successfully!")

        return output_path


def convert_process_file(
    process_name: str,
    input_path: str,
    output_dir: str = "./output"
) -> str:
    """
    Convert a single process Excel file to CSV.

    Args:
        process_name (str): Name of the process (e.g., 'laser', 'plasma')
        input_path (str): Path to input Excel file
        output_dir (str): Directory for output CSV

    Returns:
        str: Path to output CSV file
    """
    if process_name not in PROCESS_CONFIGS:
        raise ValueError(f"Unknown process: {process_name}. Available: {list(PROCESS_CONFIGS.keys())}")

    config = PROCESS_CONFIGS[process_name]
    converter = ExcelToCSVConverter(config, input_path, output_dir)
    return converter.convert()


def convert_excel_auto(
    input_path: str,
    output_dir: str = "./output"
) -> str:
    """
    Automatically detect process type and convert Excel to CSV.

    This function automatically detects which process configuration to use
    by analyzing the Excel file columns, then converts it to CSV.

    Args:
        input_path (str): Path to input Excel or CSV file
        output_dir (str): Directory for output CSV

    Returns:
        str: Path to output CSV file

    Example:
        >>> # No need to specify process name!
        >>> csv_path = convert_excel_auto('unknown_data.xlsx', './output')
        >>> print(f"Detected and converted to: {csv_path}")
    """
    logger.info(f"\n{'='*60}")
    logger.info("AUTO-DETECTION MODE")
    logger.info(f"{'='*60}")

    # Auto-detect process type
    process_name = auto_detect_process(input_path)

    logger.info(f"\nUsing configuration for: {process_name}")
    logger.info(f"{'='*60}\n")

    # Convert using detected process
    return convert_process_file(process_name, input_path, output_dir)


def convert_all_processes(
    input_files: Dict[str, str],
    output_dir: str = "./output"
) -> Dict[str, str]:
    """
    Convert multiple process files at once.

    Args:
        input_files (dict): Dictionary mapping process names to file paths
                           e.g., {'laser': '/path/to/laser.xlsx', 'plasma': '/path/to/plasma.xlsx'}
        output_dir (str): Directory for output CSV files

    Returns:
        dict: Dictionary mapping process names to output file paths
    """
    results = {}

    for process_name, input_path in input_files.items():
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {process_name}")
            logger.info(f"{'='*60}")

            output_path = convert_process_file(process_name, input_path, output_dir)
            results[process_name] = output_path

        except Exception as e:
            logger.error(f"Failed to process {process_name}: {e}")
            results[process_name] = None

    return results


def main():
    """
    Example usage of the converter.

    To use this script:
    1. Update the INPUT_FILES dictionary with your actual file paths
    2. Run: python excel_to_csv_converter.py
    """

    # Example configuration - UPDATE THESE PATHS
    INPUT_FILES = {
        # 'laser': '/path/to/your/laser.xlsx',
        # 'plasma': '/path/to/your/plasma.xlsx',
        # 'galvanic': '/path/to/your/galvanic.xlsx',
        # 'multibond': '/path/to/your/multibond.xlsx',
        # 'microetch': '/path/to/your/microetch.xlsx',
    }

    OUTPUT_DIR = './processed_data'

    if not INPUT_FILES:
        logger.warning("No input files configured. Please update INPUT_FILES in the main() function.")
        logger.info("\nExample usage:")
        logger.info("  from excel_to_csv_converter import convert_process_file")
        logger.info("  convert_process_file('laser', '/path/to/laser.xlsx', './output')")
        return

    # Convert all files
    results = convert_all_processes(INPUT_FILES, OUTPUT_DIR)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("CONVERSION SUMMARY")
    logger.info(f"{'='*60}")

    for process_name, output_path in results.items():
        status = "✓ SUCCESS" if output_path else "✗ FAILED"
        logger.info(f"{process_name:15} {status:15} {output_path or 'N/A'}")


if __name__ == "__main__":
    main()
