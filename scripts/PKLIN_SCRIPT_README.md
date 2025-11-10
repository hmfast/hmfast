# P(k) Linear Timing Script Documentation

## Overview

The `plot_pklin_timing.py` script demonstrates high-performance P(k) linear calculations using hmfast and provides comprehensive timing analysis across different cosmological parameters and redshifts.

## Key Features

### ⚡ Ultra-Fast Performance
- **Average calculation time**: ~0.8ms per P(k) spectrum
- **JAX-accelerated**: JIT compilation for optimal speed  
- **Consistent performance**: Low variance across parameter space
- **EDE-v2 emulator range**: 5×10⁻⁴ - 10.0 h/Mpc (classy_szfast compatible)

### 📊 Comprehensive Analysis
- Tests 10 different cosmological parameter sets
- Multiple redshift analysis (z = 0.0, 0.5, 1.0, 2.0)
- Statistical timing analysis with mean, std deviation, and ranges
- Visual parameter correlation analysis

### 🎯 Direct P(k) Method
```python
# UNAMBIGUOUS: Direct plotting of P(k) values - NO conversion needed!
pk, k = emulator.calculate_pkl_at_z(z=0.0, params)
plt.loglog(k, pk)  # Direct plotting with correct classy_szfast scaling
plt.ylabel('P(k) [(Mpc/h)³]')  # Includes k^(-3) scaling factor
```

## Script Structure

### Parameter Generation
- **fEDE range**: 0.001 → 0.121 (full EDE parameter space)
- **H0 range**: 65 → 75 km/s/Mpc
- **ωcdm range**: 0.10 → 0.15
- **n_s range**: 0.94 → 0.98
- **ln10¹⁰A_s range**: 2.9 → 3.1
- **Additional variations**: ωb, τ_reio with random perturbations

### Timing Methodology
1. **JIT warm-up**: First calculation excluded from timing
2. **High-precision timing**: `time.perf_counter()` for microsecond accuracy
3. **Statistical analysis**: Mean, std deviation, min/max across parameter sets
4. **Multiple redshifts**: Performance consistency across cosmic time

### Output Plots
- **Main P(k) plot**: Log-log scale showing parameter variations
- **Timing histogram**: Distribution of calculation times
- **Parameter correlation**: Time vs fEDE parameter
- **Performance summary**: Key statistics and method info

## Usage

```bash
# Activate environment and run
source hmenv/bin/activate
python scripts/plot_pklin_timing.py

# Or with custom data path
PATH_TO_CLASS_SZ_DATA=/path/to/data python scripts/plot_pklin_timing.py
```

## Expected Output

```
============================================================
hmfast P(k) Linear Timing Analysis
============================================================
✓ Successfully loaded EDE emulator
✓ Generated 10 parameter sets
  fEDE range: 0.001 - 0.121
  H0 range: 65.0 - 75.0

========================================
Testing at redshift z = 0.0
========================================
Timing P(k) linear calculations at z=0.0 for 10 parameter sets...
Warming up JIT compilation...
  Parameter set  1:   0.56 ms (fEDE=0.001)
  Parameter set  2:   0.45 ms (fEDE=0.014)
  ...
  Parameter set 10:   0.47 ms (fEDE=0.121)

Timing Statistics:
  Mean: 0.51 ± 0.07 ms
  Range: 0.44 - 0.69 ms

[Similar output for z = 0.5, 1.0, 2.0]

============================================================
P(k) Linear Timing Analysis Complete!

Summary across all redshifts:
  z=0.0: 0.51 ± 0.07 ms
  z=0.5: 1.16 ± 0.78 ms  
  z=1.0: 0.72 ± 0.12 ms
  z=2.0: 0.70 ± 0.11 ms

Overall average: 0.77 ± 0.46 ms
Total calculations: 40
============================================================
```

## Performance Benchmarks

### Speed Comparison
- **P(k) linear**: ~0.8ms (this script)
- **cl^TT calculation**: ~7.9ms (plot_cltt_timing.py)  
- **Speed ratio**: P(k) is ~10x faster than CMB spectra

### Scientific Validation
- **Physical scaling**: P(k) decreases with redshift as expected
- **Parameter variations**: Smooth trends with fEDE, H0, etc.
- **k-range coverage**: 5×10⁻⁴ - 10 h/Mpc (EDE-v2 emulator precision)
- **No extrapolation**: Strict validation within training range

## Files Generated

```
plots/
├── pklin_timing_analysis_z0.0.png    # z=0 analysis
├── pklin_timing_analysis_z0.5.png    # z=0.5 analysis  
├── pklin_timing_analysis_z1.0.png    # z=1 analysis
└── pklin_timing_analysis_z2.0.png    # z=2 analysis
```

## Key Improvements over Previous Implementation

1. **Correct P(k) scaling**: Matches classy_szfast with k^(-3) factor (EDE-v2)
2. **Linear scale output**: Returns actual P(k) in (Mpc/h)³ units  
3. **Dual naming support**: Both `get_pkl_at_z()` and `calculate_pkl_at_z()`
4. **Correct k-range**: 5×10⁻⁴ - 10.0 h/Mpc (classy_szfast EDE-v2 compatible)
5. **No extrapolation**: Clear error handling for z > z_max
6. **Enhanced performance**: JAX optimization throughout
7. **Comprehensive testing**: Multiple redshifts and parameter variations

## Integration

This script complements the existing `plot_cltt_timing.py` script, providing:
- Complete emulator performance characterization
- Cross-validation of different hmfast methods  
- Benchmarking data for optimization efforts
- Documentation of best practices for P(k) calculations

## Dependencies

- hmfast (with EDE emulator data)
- JAX (for high-performance calculations)
- matplotlib (for plotting)
- numpy (for numerical operations)
- PATH_TO_CLASS_SZ_DATA environment variable