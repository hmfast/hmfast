# cl^TT Standalone Script

## Overview
`plot_cltt_timing.py` is a clean, standalone script that demonstrates the **unambiguous cl^TT plotting method** and benchmarks performance across different cosmological parameters.

## Features
- **Unambiguous plotting**: Uses `get_cmb_spectra()` for direct plotting (no factors needed!)
- **Performance timing**: Benchmarks cl^TT calculation across ~10 different cosmologies
- **Parameter space exploration**: Varies key parameters (fEDE, H0, ωcdm, ns, As)
- **Comprehensive visualization**: Multiple plot types showing spectra and timing results

## Usage

```bash
# Activate the environment
source hmenv/bin/activate

# Run the script
python scripts/plot_cltt_timing.py
```

## Output
The script generates:
1. `cltt_timing_analysis.png` - 4-panel comprehensive analysis
2. `cltt_representative_spectra.png` - Clean comparison of representative spectra

## Performance Results
- **Average computation time**: ~7.9 ms per cl^TT spectrum
- **Assessment**: Excellent performance (< 10ms)
- **Parameter range**: 
  - fEDE: 0.001 to 0.121
  - H0: 65 to 75 km/s/Mpc

## Key Features

### Unambiguous Method
```python
# CORRECT - No ambiguity!
cmb_spectra = emulator.get_cmb_spectra(params, lmax=2500)
plt.plot(cmb_spectra['ell'], cmb_spectra['tt'])  # Direct plotting
plt.ylabel('ℓ(ℓ+1)C_ℓ^TT/(2π) [μK²]')
```

### Timing Analysis
- JIT compilation handled automatically
- Individual timing for each parameter set
- Statistical analysis (mean, std, min, max)
- Performance assessment

### Visualization
- All cl^TT spectra overlaid
- Relative differences to reference
- Timing bar chart with parameter labels
- Parameter space coverage plot

This script demonstrates that the hmfast cl^TT implementation is both **unambiguous** and **fast**.