#!/bin/bash

# Script to run EDE-v2 emulator plotting with proper environment setup
# Usage: ./run_ede_plots.sh

set -e  # Exit on any error

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HMFAST_DIR="$(dirname "$SCRIPT_DIR")"

echo "HMFast EDE-v2 Emulator Plotting Script"
echo "======================================"

# Check if we're in the hmfast directory
if [ ! -f "$HMFAST_DIR/pyproject.toml" ]; then
    echo "Error: Not in hmfast directory. Please run from hmfast root or scripts directory."
    exit 1
fi

# Change to hmfast directory
cd "$HMFAST_DIR"

# Check if virtual environment exists
if [ ! -d "hmenv" ]; then
    echo "Error: Virtual environment 'hmenv' not found."
    echo "Please create it first with: python -m venv hmenv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source hmenv/bin/activate

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import jax; print(f'JAX version: {jax.__version__}')" || {
    echo "Installing JAX..."
    pip install jax jaxlib
}

python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')" || {
    echo "Installing matplotlib..."
    pip install matplotlib
}

python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" || {
    echo "Installing numpy..."
    pip install numpy
}

# Install hmfast in development mode
echo "Installing hmfast in development mode..."
pip install -e .

# Check if PATH_TO_CLASS_SZ_DATA is set
if [ -z "$PATH_TO_CLASS_SZ_DATA" ]; then
    echo "Warning: PATH_TO_CLASS_SZ_DATA environment variable not set."
    echo "Please set it to your emulator data directory:"
    echo "  export PATH_TO_CLASS_SZ_DATA=/path/to/your/data"
    echo ""
    echo "Attempting to find data directory..."
    
    # Common locations to check
    POSSIBLE_PATHS=(
        "$HOME/class_sz_data_directory"
        "$HOME/cosmopower_data" 
        "/usr/local/share/class_sz_data"
        "$(pwd)/../class_sz_data"
    )
    
    for path in "${POSSIBLE_PATHS[@]}"; do
        if [ -d "$path/ede-v2" ]; then
            export PATH_TO_CLASS_SZ_DATA="$path"
            echo "Found data directory: $PATH_TO_CLASS_SZ_DATA"
            break
        fi
    done
    
    if [ -z "$PATH_TO_CLASS_SZ_DATA" ]; then
        echo "Error: Could not find emulator data directory."
        echo "Please set PATH_TO_CLASS_SZ_DATA manually."
        exit 1
    fi
fi

echo "Using data directory: $PATH_TO_CLASS_SZ_DATA"

# Create plots directory
mkdir -p plots

# Run the plotting script
echo ""
echo "Running EDE-v2 emulator plots..."
python scripts/ede_plots.py

echo ""
echo "✓ Plotting completed successfully!"
echo "✓ Plots saved to: $(pwd)/plots/"
echo "✓ Virtual environment: hmenv"

# List generated plots
echo ""
echo "Generated plots:"
ls -la plots/*.png 2>/dev/null || echo "No PNG files found"
ls -la plots/*.pdf 2>/dev/null || echo "No PDF files found"