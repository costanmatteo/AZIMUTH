# Excel to CSV Converter

A Python tool for converting manufacturing process data from Excel files to structured CSV files suitable for neural network training.

## Supported Processes

The converter supports five manufacturing processes:

1. **Laser** - Laser processing data
2. **Plasma** - Plasma processing data
3. **Galvanic** - Galvanic processing data
4. **Multibond** - Multibond processing data
5. **Microetch** - Microetch processing data

## Features

- **Automatic column extraction**: Only extracts relevant columns based on process configuration
- **Date parsing**: Handles different date formats for each process
- **Data cleaning**: Removes empty rows and handles missing values
- **Flexible input**: Supports both Excel (.xlsx, .xls) and CSV input files
- **Batch processing**: Convert multiple files at once
- **Detailed logging**: Track the conversion process with informative logs

## Installation

Make sure you have the required dependencies:

```bash
pip install pandas openpyxl
```

## Quick Start

### Option 1: Convert a Single File

```python
from excel_to_csv_converter import convert_process_file

# Convert laser data
output_path = convert_process_file(
    process_name='laser',
    input_path='/path/to/your/laser.xlsx',
    output_dir='./processed_data'
)

print(f"Converted file saved to: {output_path}")
```

### Option 2: Convert Multiple Files

```python
from excel_to_csv_converter import convert_all_processes

input_files = {
    'laser': '/path/to/laser.xlsx',
    'plasma': '/path/to/plasma.xlsx',
    'galvanic': '/path/to/galvanic.xlsx',
    'multibond': '/path/to/multibond.xlsx',
    'microetch': '/path/to/microetch.xlsx',
}

results = convert_all_processes(input_files, output_dir='./processed_data')
```

### Option 3: Use the Example Script

1. Open `convert_example.py`
2. Update the file paths
3. Run:

```bash
python convert_example.py
```

## Process Configurations

Each process has specific column mappings and date formats:

### Laser
- **Input columns**: Laser, Process_1, Machine, WA, PanelNr, PaPosNr, TimeStamp/CreateDate 1
- **Output file**: `laser.csv`
- **Date format**: `%m/%d/%y %I:%M %p`

### Plasma
- **Input columns**: Plasma, Process_2, Machine, WA, PanelNummer, Position, Buchungsdatum
- **Output file**: `plasma_fixed.csv`
- **Date format**: `%m/%d/%y %I:%M %p`

### Galvanic
- **Input columns**: Galvanic, Process_3, WA, Panelnr, PaPosNr, Date/Time Stamp
- **Output file**: `galvanik.csv`
- **Date format**: `%m/%d/%y %I:%M %p`

### Multibond
- **Input columns**: Multibond, Process_4, WA, PaPosNr, t_StartDateTime
- **Output file**: `multibond.csv`
- **Date format**: `%m/%d/%y %I:%M %p`

### Microetch
- **Input columns**: Microetch, Process_5, WA, PaPosNr, CreateDate
- **Output file**: `microetch.csv`
- **Date format**: `%d.%m.%Y %H:%M:%S`

## Output Format

The converter creates CSV files with standardized column names:

```
{prefix}_process        - Process name
{prefix}_hidden_process - Hidden process identifier
{prefix}_machine        - Machine identifier (if available)
{prefix}_wa             - Work Area
{prefix}_panel          - Panel number (if available)
{prefix}_papos          - Panel position number
{prefix}_timestamp      - Timestamp of the operation
```

Where `{prefix}` is:
- `las` for Laser
- `pla` for Plasma
- `gal` for Galvanic
- `mul` for Multibond
- `mic` for Microetch

## Example Output

Input Excel with columns: `TimeStamp, Machine, WA, PanelNr, PaPosNr, Laser, Process_1, ExtraColumn`

Output CSV will contain only: `las_timestamp, las_machine, las_wa, las_panel, las_papos, las_process, las_hidden_process`

## Advanced Usage

### Custom Configuration

You can create your own process configuration:

```python
from excel_to_csv_converter import ExcelToCSVConverter

custom_config = {
    "process_label": "MyProcess",
    "hidden_label": "Process_ID",
    "machine_label": "MachineID",
    "WA_label": "WorkArea",
    "panel_label": "Panel",
    "PaPos_label": "Position",
    "date_label": ["Timestamp"],
    "date_format": "%Y-%m-%d %H:%M:%S",
    "prefix": "custom",
    "filename": "custom_process.csv",
    "sep": ",",
    "header": 0
}

converter = ExcelToCSVConverter(
    config=custom_config,
    input_path='/path/to/input.xlsx',
    output_dir='./output'
)

output_path = converter.convert()
```

### Handling Missing Columns

The converter will:
- Skip columns that don't exist in the input file
- Log warnings for missing columns
- Continue processing with available columns
- Remove rows with missing critical identifiers (like WA)

### Date Parsing

If the specified date format doesn't work, the converter will:
1. Try automatic date detection
2. Log warnings for unparseable dates
3. Continue processing (dates will be null for failed parses)

## Troubleshooting

### "No relevant columns found"
- Check that your Excel file has the expected column names
- Column names are case-sensitive
- Use `example_check_columns()` to see expected column names

### "Error loading file"
- Verify the file path is correct
- Ensure the file is not open in Excel
- Check file permissions

### Date parsing issues
- Verify the date format in your Excel matches the configuration
- Dates will be set to null if they can't be parsed
- Check the logs for specific parsing errors

## Integration with Neural Network Pipeline

After conversion, use the CSV files with the existing preprocessing pipeline:

```python
from data.preprocessing import load_csv_data, DataPreprocessor

# Load converted data
X, y = load_csv_data(
    filepath='./processed_data/laser.csv',
    input_columns=['las_machine', 'las_wa', 'las_panel'],
    output_columns=['las_papos']
)

# Preprocess for training
preprocessor = DataPreprocessor(scaling_method='standard')
X_scaled, y_scaled = preprocessor.fit_transform(X, y)
```

## License

This tool is part of the AZIMUTH project.
