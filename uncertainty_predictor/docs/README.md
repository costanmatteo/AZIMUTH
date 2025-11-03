# SCM Documentation

This directory contains comprehensive technical documentation for the Structural Causal Model (SCM) framework integrated into the AZIMUTH uncertainty quantification system.

## Files

- **scm_documentation.tex**: Complete LaTeX documentation covering:
  - Mathematical foundations of SCMs
  - Implementation architecture
  - Two dataset implementations (parent-child and laser power models)
  - Integration with the uncertainty predictor
  - Usage guide and examples
  - Troubleshooting and best practices

## Compiling the Documentation

### Requirements

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages: amsmath, tikz, listings, algorithm, hyperref, etc.

### Compilation

```bash
cd uncertainty_predictor/docs
pdflatex scm_documentation.tex
pdflatex scm_documentation.tex  # Run twice for references
```

Or use:

```bash
latexmk -pdf scm_documentation.tex
```

### Output

The compilation will generate `scm_documentation.pdf`.

## Quick Preview

The documentation includes:

1. **Introduction**: Motivation and system overview
2. **Mathematical Foundations**: Formal SCM definitions, DAG constraints, sampling algorithms
3. **Implementation Architecture**: Core classes (NodeSpec, NoiseModel, SCM, SCMDataset)
4. **Dataset Implementations**:
   - Parent-Child with Cross-Talk (10 inputs → 1 output)
   - Laser Actual Power (2 inputs → 1 output, physics-based)
5. **Integration Guide**: How SCM connects to the training pipeline
6. **Usage Examples**: Configuration, custom datasets, interventional queries
7. **Troubleshooting**: Common errors and solutions
8. **Appendix**: Complete code examples

## Quick Start

To use the SCM framework without reading the full documentation:

1. Edit `configs/example_config.py`:
   ```python
   CONFIG = {
       'data': {
           'csv_path': None,
           'use_scm': True,
           'scm': {
               'n_samples': 5000,
               'seed': 42,
               'dataset_type': 'one_to_one_ct'  # or 'laser'
           }
       }
   }
   ```

2. Run training:
   ```bash
   cd uncertainty_predictor
   python train.py
   ```

See the full documentation for advanced features and custom dataset creation.
