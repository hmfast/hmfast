# HMFast EDE-v2 Emulator Scripts

This directory contains scripts for running and plotting with the EDE-v2 emulator.

## Quick Start

To run the complete plotting suite with proper environment setup:

```bash
./run_ede_plots.sh
```

This script will:
1. ✅ Activate the hmfast virtual environment (`source hmenv/bin/activate`)
2. ✅ Check and install required dependencies
3. ✅ Install hmfast in development mode
4. ✅ Verify the emulator data directory
5. ✅ Run all plotting routines
6. ✅ Save high-quality plots to the `plots/` directory

## Generated Plots

The script creates the following publication-quality plots:

### 1. **angular_distances.png**
- Angular diameter distance evolution vs redshift
- Comparison between EDE and ΛCDM models
- Relative differences panel

### 2. **hubble_evolution.png** 
- Hubble parameter H(z) evolution
- EDE vs ΛCDM comparison
- Shows the impact of early dark energy on expansion

### 3. **power_spectra.png**
- Linear and nonlinear power spectra P(k)
- Multiple redshifts (z = 0, 1)
- Relative difference analysis

### 4. **sigma8_evolution.png**
- σ₈ evolution with redshift
- Structure growth comparison
- Current values highlighted

### 5. **cmb_spectra.png**
- Complete CMB power spectra suite:
  - TT (temperature)
  - EE (E-mode polarization) 
  - TE (temperature-E cross-correlation)
  - PP (lensing potential)

### 6. **derived_parameters.png**
- Comparison of key derived parameters
- Bar charts with precise values
- Relative differences quantified

### 7. **performance_benchmark.png**
- Performance scaling analysis
- JIT compilation speedup factors
- Method timing comparisons
- Memory usage estimates

### 8. **parameter_sensitivity.png**
- Sensitivity to EDE parameters:
  - fEDE (EDE fraction)
  - log₁₀z_c (characteristic redshift)
  - H₀ (Hubble constant)

### 9. **summary_plot.png**
- Comprehensive 12-panel summary
- All key quantities in one figure
- Perfect for presentations/papers

## Requirements

### Environment Setup
The script automatically handles environment setup, but you need:

1. **Virtual Environment**: 
   ```bash
   python -m venv hmenv  # Create if doesn't exist
   ```

2. **Data Directory**: Set the environment variable:
   ```bash
   export PATH_TO_CLASS_SZ_DATA=/path/to/your/emulator/data
   ```

### Dependencies
Auto-installed by the script:
- `jax` and `jaxlib` (JAX for acceleration)
- `matplotlib` (plotting)
- `numpy` (numerical arrays)
- `hmfast` (the emulator package)

## Manual Usage

If you want to run components manually:

```bash
# Activate environment
source hmenv/bin/activate

# Install hmfast
pip install -e .

# Run plotting script directly
python scripts/ede_plots.py
```

## Customization

### Modify Parameters
Edit the `define_cosmologies()` function in `ede_plots.py`:

```python
ede_params = {
    'omega_b': 0.02242,
    'omega_cdm': 0.11933,
    'H0': 67.66,
    'fEDE': 0.087,        # Modify this
    'log10z_c': 3.562,    # Or this
    'thetai_scf': 2.83,   # Or this
}
```

### Add New Plots
Add functions to `ede_plots.py`:

```python
def plot_my_analysis(emulator, params, save_dir):
    # Your custom analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get emulator predictions
    result = emulator.get_some_quantity(params)
    
    # Make plots
    ax.plot(...)
    
    plt.savefig(f'{save_dir}/my_analysis.png')
    plt.close()

# Add to main() function
plot_my_analysis(emulator, ede_params, save_dir)
```

### Output Formats
Change output format by modifying the `plt.savefig()` calls:

```python
plt.savefig(f'{save_dir}/plot.pdf')  # PDF format
plt.savefig(f'{save_dir}/plot.svg')  # SVG format  
plt.savefig(f'{save_dir}/plot.eps')  # EPS format
```

## Performance Notes

- **First run**: ~30-60 seconds (includes JAX compilation)
- **Subsequent runs**: ~10-20 seconds (using compiled code)
- **Memory usage**: ~100-500 MB peak
- **Plots quality**: 300 DPI, publication ready

## Troubleshooting

### Common Issues

1. **"PATH_TO_CLASS_SZ_DATA not set"**
   ```bash
   export PATH_TO_CLASS_SZ_DATA=/your/data/path
   ```

2. **"Virtual environment not found"**
   ```bash
   python -m venv hmenv
   ```

3. **"EDE-v2 emulator data not found"**
   - Verify `ede-v2/` directory exists in data path
   - Check that `.npz` files are present in subdirectories

4. **Import errors**
   ```bash
   source hmenv/bin/activate
   pip install -e .
   ```

5. **JAX/GPU issues**
   ```bash
   # Force CPU mode if needed
   export JAX_PLATFORM_NAME=cpu
   ```

### Debug Mode
For detailed error information, modify the script to add:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Output Directory

All plots are saved to `plots/` with:
- **High resolution**: 300 DPI
- **Clean formatting**: Tight bounding boxes
- **Consistent styling**: Professional appearance
- **Multiple formats**: Add PDF/SVG as needed

## Integration with Notebooks

These scripts reproduce and extend the key analyses from the CLASS-SZ notebooks:

- `classy_szfast_angular_distance_gpu.ipynb`
- `classy_szfast_parameters_and_derived_parameters.ipynb`
- `classy_szfast_matter_pk_linear.ipynb`
- `classy_szfast_cmb_cls.ipynb`

The advantage is:
- ✅ **Reproducible**: Version controlled scripts
- ✅ **Automated**: No manual cell execution
- ✅ **Scalable**: Easy to run parameter sweeps
- ✅ **Publication ready**: High-quality output formats

## Citation

If you use these plots in publications, please cite the hmfast package and the underlying CLASS-SZ emulators.