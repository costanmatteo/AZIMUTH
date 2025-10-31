# Process-Based Training Guide

This guide explains how to use the automatic process-based configuration system for training uncertainty quantification models.

## Overview

Instead of manually specifying input and output columns for each training session, you can now:

1. **Select a process type** (laser, plasma, galvanic, multibond, microetch)
2. **Define output columns** (what the model should predict)
3. **Automatic column mapping** - the system automatically:
   - Loads the correct CSV file
   - Excludes metadata columns (timestamps, IDs, machine labels, etc.)
   - Uses all remaining columns as inputs

## Quick Start

### Step 1: Configure Output Columns

Edit `configs/process_config.py` and set the output columns for your process:

```python
PROCESS_CONFIGS = {
    "laser": {
        # ... other config ...
        "output_columns": [
            "Temperature",      # Example target variable
            "Quality_Score",    # Example target variable
        ],
    },
}
```

### Step 2: Load Data Automatically

```python
from src.data import load_process_data

# Load laser process data with automatic column mapping
X, y, column_info = load_process_data('laser')

# See what columns were selected
print("Inputs:", column_info['input_columns'])
print("Outputs:", column_info['output_columns'])
print("Excluded:", column_info['metadata_columns'])
```

### Step 3: Train Your Model

Use the example script:

```bash
python examples/train_with_process.py
```

Or use the process-based config:

```bash
python train.py --config configs/process_based_config.py
```

## Available Processes

| Process | CSV File | Prefix | Metadata Columns |
|---------|----------|--------|------------------|
| `laser` | laser.csv | las | WA, PanelNr, PaPosNr, TimeStamp, CreateDate 1, Process_1, Machine |
| `plasma` | plasma_fixed.csv | pla | WA, PanelNummer, Position, Buchungsdatum, Process_2, Machine |
| `galvanic` | galvanik.csv | gal | WA, Panelnr, PaPosNr, Date/Time Stamp, Process_3 |
| `multibond` | multibond.csv | mul | WA, PaPosNr, t_StartDateTime, Process_4 |
| `microetch` | microetch.csv | mic | WA, PaPosNr, CreateDate, Process_5 |

## Column Selection Logic

For each process, columns are categorized as:

1. **Metadata columns** - Excluded from training (defined in `process_config.py`)
   - Identifiers: WA, Panel numbers, Position numbers
   - Timestamps: Date/time fields
   - Process labels: Hidden process labels, machine identifiers

2. **Output columns** - Target variables to predict (you define these)
   - Quality metrics
   - Measurements
   - Process outcomes

3. **Input columns** - Everything else (automatically determined)
   - All columns not in metadata or output categories
   - Process parameters
   - Sensor readings
   - Operating conditions

## Configuration Methods

### Method 1: Edit process_config.py (Recommended)

Permanently set output columns for a process:

```python
# In configs/process_config.py
PROCESS_CONFIGS = {
    "laser": {
        "output_columns": ["Temperature", "Quality"],
    },
}
```

### Method 2: Runtime Override

Specify output columns when loading data:

```python
X, y, info = load_process_data(
    'laser',
    output_columns=['Temperature', 'Quality']
)
```

### Method 3: Using process_based_config.py

Set process and outputs in the config file:

```python
# In configs/process_based_config.py
PROCESS_TYPE = 'laser'
OUTPUT_COLUMNS = ['Temperature', 'Quality']
```

## Finding Available Columns

If you're unsure which columns are available, run with empty output columns:

```python
# This will show all available columns
X, y, info = load_process_data('laser', output_columns=[])
```

The error message will display:
- All columns in the CSV
- Which are metadata (excluded)
- Which are available for input/output

## Example Workflow

### 1. Explore Available Columns

```python
from src.data import load_process_data

# Try loading without output columns to see what's available
try:
    X, y, info = load_process_data('laser')
except ValueError as e:
    print(e)  # Shows available columns
```

### 2. Define Your Targets

```python
# Load with specific output columns
X, y, info = load_process_data(
    process_name='laser',
    data_dir='src/data/raw',
    output_columns=['TargetTemp', 'QualityScore']
)

print(f"Loaded {X.shape[0]} samples")
print(f"Input features: {len(info['input_columns'])}")
print(f"Output features: {len(info['output_columns'])}")
```

### 3. Preprocess and Train

```python
from src.data import DataPreprocessor

# Preprocess
preprocessor = DataPreprocessor(scaling_method='standard')
X_scaled, y_scaled = preprocessor.fit_transform(X, y)

# Split
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
    X_scaled, y_scaled
)

# Create model and train...
```

## Advanced Usage

### Custom Process Configuration

Add a new process to `configs/process_config.py`:

```python
PROCESS_CONFIGS = {
    "my_process": {
        "process_label": "MyProcess",
        "hidden_label": "Process_6",
        "WA_label": "WorkArea",
        "panel_label": "Panel",
        "date_label": ["Timestamp"],
        "date_format": "%Y-%m-%d %H:%M:%S",
        "prefix": "myp",
        "filename": "my_process.csv",
        "sep": ",",
        "header": 0,
        "metadata_columns": [
            "WorkArea", "Panel", "Timestamp", "Process_6"
        ],
        "output_columns": [
            "MyTarget"
        ],
    }
}
```

### Programmatic Configuration

```python
from configs.process_config import (
    get_process_config,
    set_output_columns,
    get_available_processes
)

# See available processes
print("Available:", get_available_processes())

# Set outputs for a process
set_output_columns('laser', ['Temp1', 'Temp2'])

# Get full config
config = get_process_config('laser')
print(config)
```

## Benefits

✅ **No manual column selection** - Automatically excludes metadata
✅ **Consistent configuration** - Process settings centralized
✅ **Reusable** - Same config for training and inference
✅ **Flexible** - Override outputs per experiment
✅ **Self-documenting** - Column mapping is explicit and saved

## Troubleshooting

### FileNotFoundError

```
FileNotFoundError: CSV file not found: src/data/raw/laser.csv
```

**Solution:** Make sure the CSV file exists in the specified `data_dir`

### ValueError: No output columns defined

```
ValueError: No output columns defined for process 'laser'
```

**Solution:** Define output columns using one of the three methods above

### ValueError: Unknown process

```
ValueError: Unknown process 'lasers'. Available processes: laser, plasma, ...
```

**Solution:** Check the process name spelling. Use `get_available_processes()` to see valid options

### KeyError: Column not found

```
KeyError: "['Temperature'] not found in axis"
```

**Solution:** The column doesn't exist in the CSV. Check column names match exactly (case-sensitive)

## See Also

- `examples/train_with_process.py` - Complete training example
- `configs/process_based_config.py` - Configuration template
- `configs/process_config.py` - Process definitions
- `src/data/preprocessing.py` - Data loading functions
