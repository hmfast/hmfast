#!/usr/bin/env python3
"""
Test JAX compilation behavior with different cosmological/GNFW parameters.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import time
jax.config.update("jax_enable_x64", True)

# Set up environment
if 'PATH_TO_CLASS_SZ_DATA' not in os.environ:
    os.environ['PATH_TO_CLASS_SZ_DATA'] = '/Users/boris/class_sz_data_directory'

import sys
sys.path.append('../src')
from hmfast.tsz_power import TSZPowerSpectrum

def main():
    print("=" * 60)
    print("Testing Parameter Changes vs JAX Compilation")
    print("=" * 60)
    
    # Initialize TSZ calculator
    tsz_calc = TSZPowerSpectrum()
    print("✓ TSZPowerSpectrum initialized")
    
    # Base parameters
    base_params = {
        'omega_b': 0.02242,
        'omega_cdm': 0.11933,
        'H0': 67.66,
        'tau_reio': 0.0561,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665,
        'B': 1.0,
        'c500': 1.81,
        'gammaGNFW': 0.31,
        'alphaGNFW': 1.33,
        'betaGNFW': 4.13,
        'P0GNFW': 6.41,
    }
    
    # Test grid settings (small for speed)
    grid_settings = {
        'lmin': 10, 'lmax': 1000, 'dlogell': 0.2,
        'n_z': 20, 'n_m': 20
    }
    
    print("\\nTest 1: Initial compilation")
    print("-" * 40)
    start = time.time()
    ell1, C_yy1 = tsz_calc.compute_clyy_fast(base_params, **grid_settings)
    initial_time = time.time() - start
    print(f"Initial call (with compilation): {initial_time:.3f}s")
    
    print("\\nTest 2: Same parameters (compiled)")
    print("-" * 40)
    start = time.time()
    ell2, C_yy2 = tsz_calc.compute_clyy_fast(base_params, **grid_settings)
    same_time = time.time() - start
    print(f"Same parameters: {same_time:.3f}s")
    
    print("\\nTest 3: Different cosmological parameters")
    print("-" * 40)
    
    cosmo_variations = [
        {"name": "Higher H0", "H0": 70.0},
        {"name": "Higher Ωcdm", "omega_cdm": 0.13},
        {"name": "Higher σ8", "ln10^{10}A_s": 3.1},
        {"name": "Different ns", "n_s": 0.96},
    ]
    
    cosmo_times = []
    for variation in cosmo_variations:
        test_params = base_params.copy()
        test_params.update({k: v for k, v in variation.items() if k != "name"})
        
        start = time.time()
        ell, C_yy = tsz_calc.compute_clyy_fast(test_params, **grid_settings)
        elapsed = time.time() - start
        cosmo_times.append(elapsed)
        
        print(f"  {variation['name']}: {elapsed:.3f}s")
    
    print("\\nTest 4: Different GNFW parameters")
    print("-" * 40)
    
    gnfw_variations = [
        {"name": "Higher c500", "c500": 2.5},
        {"name": "Different γ", "gammaGNFW": 0.4},
        {"name": "Different α", "alphaGNFW": 1.5},
        {"name": "Different β", "betaGNFW": 4.5},
        {"name": "Different P0", "P0GNFW": 8.0},
        {"name": "Mass bias", "B": 0.8},
    ]
    
    gnfw_times = []
    for variation in gnfw_variations:
        test_params = base_params.copy()
        test_params.update({k: v for k, v in variation.items() if k != "name"})
        
        start = time.time()
        ell, C_yy = tsz_calc.compute_clyy_fast(test_params, **grid_settings)
        elapsed = time.time() - start
        gnfw_times.append(elapsed)
        
        print(f"  {variation['name']}: {elapsed:.3f}s")
    
    print("\\nTest 5: Extreme parameter changes")
    print("-" * 40)
    
    extreme_params = base_params.copy()
    extreme_params.update({
        'H0': 50.0,         # Very low H0
        'omega_cdm': 0.20,  # High dark matter
        'c500': 5.0,        # High concentration
        'P0GNFW': 15.0,     # High normalization
        'B': 0.5            # Strong mass bias
    })
    
    start = time.time()
    ell_ext, C_yy_ext = tsz_calc.compute_clyy_fast(extreme_params, **grid_settings)
    extreme_time = time.time() - start
    print(f"Extreme parameters: {extreme_time:.3f}s")
    
    print("\\nTest 6: What triggers recompilation?")
    print("-" * 40)
    
    # Test changing grid sizes (this SHOULD trigger recompilation)
    different_grid = {
        'lmin': 10, 'lmax': 1000, 'dlogell': 0.2,
        'n_z': 25, 'n_m': 25  # Different sizes
    }
    
    start = time.time()
    ell_diff, C_yy_diff = tsz_calc.compute_clyy_fast(base_params, **different_grid)
    grid_change_time = time.time() - start
    print(f"Different grid size (25×25 vs 20×20): {grid_change_time:.3f}s")
    
    # Back to original grid
    start = time.time()
    ell_back, C_yy_back = tsz_calc.compute_clyy_fast(base_params, **grid_settings)
    back_time = time.time() - start
    print(f"Back to original grid (20×20): {back_time:.3f}s")
    
    print("\\n" + "=" * 60)
    print("COMPILATION BEHAVIOR ANALYSIS")
    print("=" * 60)
    
    print(f"Initial compilation: {initial_time:.3f}s")
    print(f"Same parameters: {same_time:.3f}s")
    print(f"Speedup from compilation: {initial_time/same_time:.1f}×")
    
    avg_cosmo_time = np.mean(cosmo_times)
    avg_gnfw_time = np.mean(gnfw_times)
    
    print(f"\\nDifferent cosmology: {avg_cosmo_time:.3f}s (avg)")
    print(f"Different GNFW: {avg_gnfw_time:.3f}s (avg)")
    print(f"Extreme parameters: {extreme_time:.3f}s")
    
    print(f"\\nGrid size changes:")
    print(f"  New grid size: {grid_change_time:.3f}s")
    print(f"  Back to original: {back_time:.3f}s")
    
    # Analysis
    sub_second_threshold = 1.0
    
    print(f"\\n🎯 RESULTS:")
    if avg_cosmo_time < sub_second_threshold and avg_gnfw_time < sub_second_threshold:
        print("✅ Parameter changes do NOT trigger recompilation!")
        print("✅ Sub-second performance maintained for all parameter variations")
        print("🚀 Ready for parameter estimation and MCMC!")
    else:
        print("❌ Some parameter changes are slow")
    
    if grid_change_time > 2 * same_time:
        print("⚠️  Changing grid sizes triggers recompilation")
        print("💡 Keep consistent grid sizes for best performance")
    
    print(f"\\n📊 Performance Summary:")
    print(f"  Same params: {same_time:.3f}s")
    print(f"  Cosmo changes: {avg_cosmo_time:.3f}s ± {np.std(cosmo_times):.3f}s")
    print(f"  GNFW changes: {avg_gnfw_time:.3f}s ± {np.std(gnfw_times):.3f}s")
    print(f"  Extreme changes: {extreme_time:.3f}s")
    
    if max(avg_cosmo_time, avg_gnfw_time, extreme_time) < sub_second_threshold:
        print("\\n🎉 ALL PARAMETER CHANGES STAY SUB-SECOND!")
        print("Perfect for cosmological parameter estimation!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()