# Spectrum Analysis Software

A professional software tool for spectral data preprocessing, analysis, and modeling.

## Features

- **Supports Multiple Spectrum Types**: Near-infrared (NIR), mid-infrared (MIR), Raman spectroscopy, etc.
- **Rich Preprocessing Methods**: Normalization, smoothing, derivation, baseline correction, etc.
- **Powerful Analysis Tools**: Multivariate statistical analysis methods such as PCA, PLS, SVM, etc.
- **Corn Component Content Prediction**: Specialized analysis module for near-infrared corn spectra
- **User-Friendly Interface**: Interactive graphical interface built with PyQt5
- **Flexible Data Management**: Supports multiple import and export formats, session saving and restoration
- **High-Quality Visualization**: Various spectral plots, analysis result charts

## System Requirements

- Python 3.7 or higher
- Operating System: Windows/Linux/macOS

## Quick Start

### Install Dependencies

bash
pip install -r requirements.txt


### Run the Software

bash
python src/main.py


## User Guide

### Import Data

1. Click "File" -> "Open", or use the "Import" button on the toolbar
2. Select the spectral data file to import
3. Configure import parameters as needed

### Preprocess Data

1. Select the desired preprocessing method in the "Preprocessing" tab
2. Set the corresponding parameters
3. Click the "Apply" button to process
4. A comparison of results before and after preprocessing will be displayed in the graphical area

### Analyze Data

1. Switch to the "Analysis" tab
2. Select an appropriate analysis method (PCA, PLS, etc.)
3. Set analysis parameters
4. Click the "Run" button to execute the analysis
5. View result charts and data reports

### Corn Component Analysis

1. Switch to the dedicated "Corn Analysis" tab
2. Click "Load Corn Data File" to import spectral data with component content
3. Select appropriate preprocessing methods and apply them
4. Choose a model type (PLS, SVR, or ensemble model)
5. Set cross-validation parameters and click "Train Model"
6. View prediction results and evaluation metrics

### Save Results

1. Click "File" -> "Save" to save the current session
2. Click "File" -> "Export Results" to export analysis results

## Supported File Formats

- CSV (Comma-Separated Values)
- TXT (Tab-Separated Text File)
- JCAMP-DX (Standard format for spectral data)
- JSON (For metadata and settings)
- HDF5 (For large datasets)
- MAT (MATLAB data file)

## Development Guide

Detailed development documentation can be found in the `docs/` directory:

- [Software Design Document](docs/design.md)
- [Project Dependencies](docs/requirements.md)

### Project Structure


spectrum_analyzer/
├── docs/                   # Documentation
├── src/                    # Source code
│   ├── main.py             # Main program entry
│   ├── core/               # Core algorithms
│   ├── data/               # Data processing
│   ├── gui/                # Graphical interface
│   └── tests/              # Test code
├── data/                   # Sample data
├── requirements.txt        # Dependency list
└── README.md               # Project description


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or suggestions, please submit them via the [issues] page.
