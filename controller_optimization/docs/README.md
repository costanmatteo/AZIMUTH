# Controller Optimization Documentation

This directory contains comprehensive documentation for the AZIMUTH Controller Optimization system.

## Files

- `controller_optimization_conceptual_guide.tex` - Complete LaTeX source document explaining the controller optimization system from a conceptual perspective

## Compiling the LaTeX Document

To generate the PDF from the LaTeX source, you need a LaTeX distribution installed on your system.

### Prerequisites

Install a LaTeX distribution:
- **Linux**: `sudo apt-get install texlive-full` (Debian/Ubuntu) or `sudo yum install texlive` (RedHat/CentOS)
- **macOS**: Install MacTeX from https://www.tug.org/mactex/
- **Windows**: Install MiKTeX from https://miktex.org/

### Compilation

Navigate to this directory and run:

```bash
cd /path/to/AZIMUTH/controller_optimization/docs
pdflatex controller_optimization_conceptual_guide.tex
pdflatex controller_optimization_conceptual_guide.tex  # Run twice for references
```

Or use latexmk for automatic compilation:

```bash
latexmk -pdf controller_optimization_conceptual_guide.tex
```

### Output

The compilation will generate:
- `controller_optimization_conceptual_guide.pdf` - The final PDF document
- Various auxiliary files (.aux, .log, .toc, etc.) - Can be safely deleted after compilation

## Document Contents

The conceptual guide covers:

1. **Introduction** - Motivation and system overview
2. **Architecture and Components** - Detailed explanation of system components
   - Uncertainty Predictors
   - Policy Generators
   - Scenario Encoder
   - Surrogate Model
3. **Training Methodology** - Two-phase training strategy and multi-scenario learning
4. **Optimization Algorithms** - Loss functions and training algorithms
5. **Evaluation Metrics** - Primary and advanced performance metrics
6. **Conceptual Innovations** - Key design principles and innovations
7. **Visualization and Reporting** - Analysis and reporting capabilities
8. **Implementation Considerations** - Practical aspects and best practices
9. **Future Directions** - Planned enhancements and extensions
10. **Appendices** - Mathematical notation, configuration examples, file locations

## Alternative Viewing

If you cannot compile the LaTeX document locally, you can:

1. **Overleaf**: Upload the .tex file to https://www.overleaf.com for online compilation
2. **LaTeX editors**: Use editors like TeXstudio, TeXmaker, or Visual Studio Code with LaTeX Workshop extension
3. **Online converters**: Use services like https://latexbase.com or https://www.latex-to-pdf.com

## Contact

For questions about the controller optimization system or this documentation, please refer to the main project README or open an issue in the repository.
