
# hmfast

Machine learning accelerated cosmological emulator with **unambiguous cl^TT plotting**.

## Overview

hmfast provides fast, JAX-compatible emulated cosmological calculations, specifically:
- **EDE-v2 emulator** for Early Dark Energy models
- **Unambiguous cl^TT plotting** - no factor confusion!
- High-performance calculations (~7.9ms per spectrum)

## Quick Start

```bash
# Activate environment
source hmenv/bin/activate

# Test the setup
python scripts/test_setup.py

# Run cl^TT timing analysis (10 different cosmologies)
python scripts/plot_cltt_timing.py

# Generate comprehensive EDE plots
./scripts/run_ede_plots.sh
```

## Key Features

### ✅ Unambiguous cl^TT Method
```python
from hmfast import EDEEmulator

emulator = EDEEmulator()
cmb_spectra = emulator.get_cmb_spectra(params, lmax=2500)
plt.plot(cmb_spectra['ell'], cmb_spectra['tt'])  # Direct plotting - NO factors!
plt.ylabel('ℓ(ℓ+1)C_ℓ^TT/(2π) [μK²]')
```

### ⚡ High Performance
- Average cl^TT computation: **7.9ms**
- JAX JIT compilation for speed
- Excellent scaling across parameter space

### 🎯 Clean Codebase
- Focused on EDE emulation only
- Removed unused modules and duplicates
- Clear documentation and examples

## Repository Structure

```
hmfast/
├── src/hmfast/           # Core package
│   ├── ede_emulator.py   # Main EDE-v2 emulator
│   ├── clean_nn_emulator.py  # Neural network utilities
│   └── utils.py          # Utility functions
├── scripts/              # Analysis scripts
│   ├── plot_cltt_timing.py   # cl^TT standalone script (NEW!)
│   ├── ede_plots.py          # Comprehensive plotting
│   └── test_setup.py         # Setup verification
├── examples/             # Usage examples
├── tests/                # Unit tests
├── plots/                # Generated plots
└── convert_ede_v2.py     # Conversion script (tested & verified)
```

## Documentation

- `CMB_SPECTRA_PLOTTING_GUIDE.md` - Unambiguous plotting method
- `scripts/CLTT_SCRIPT_README.md` - cl^TT timing script details
- `EDE_EMULATOR_README.md` - EDE-v2 emulator details

## Installation

```bash
pip install -e .
```

Ensure `PATH_TO_CLASS_SZ_DATA` points to your emulator data directory.

### Converting Emulator Files

If you need to convert original TensorFlow-based emulator files to numpy format:

```bash
# Convert from ede to ede_v2_numpy
python convert_ede_v2.py

# Or specify custom paths
python convert_ede_v2.py /path/to/ede /path/to/ede_v2_numpy
```

## Performance

- **cl^TT calculation**: 7.9ms average (excellent)
- **Parameter range tested**: fEDE 0.001→0.121, H0 65→75
- **JAX-accelerated**: Fast and differentiable