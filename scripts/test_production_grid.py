#!/usr/bin/env python3
"""
Test production-quality tSZ calculation with 100×100 grid.
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
    print("Production-Quality tSZ Calculation: 100×100 Grid")
    print("=" * 60)
    
    # Initialize TSZ calculator
    tsz_calc = TSZPowerSpectrum()
    print("✓ TSZPowerSpectrum initialized")
    
    # Cosmological parameters
    cosmo_params = {
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
    
    print("\\nTest 1: Production grid with compilation")
    print("-" * 50)
    print("Grid: 100z × 100m × ell[10→10000]")
    
    start_time = time.time()
    ell, C_yy = tsz_calc.compute_clyy_fast(params_values_dict=cosmo_params)
    compilation_time = time.time() - start_time
    
    n_ell = len(ell)
    total_evals = 100 * 100 * n_ell
    
    print(f"✓ First call (with compilation): {compilation_time:.3f}s")
    print(f"✓ Grid: {n_ell} ell × 100 z × 100 masses = {total_evals:,} evaluations")
    print(f"✓ C_l range: [{C_yy.min():.2e}, {C_yy.max():.2e}]")
    
    print("\\nTest 2: Compiled performance")
    print("-" * 50)
    
    # Multiple compiled runs
    compiled_times = []
    for i in range(3):
        start_time = time.time()
        ell, C_yy = tsz_calc.compute_clyy_fast(params_values_dict=cosmo_params)
        elapsed = time.time() - start_time
        compiled_times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.3f}s")
    
    avg_compiled = np.mean(compiled_times)
    std_compiled = np.std(compiled_times)
    
    print(f"\\n✅ Average compiled time: {avg_compiled:.3f}s ± {std_compiled:.3f}s")
    print(f"✅ Speedup from compilation: {compilation_time/avg_compiled:.1f}×")
    
    # Performance analysis
    rate_ms = avg_compiled / total_evals * 1000
    print(f"✅ Rate: {rate_ms:.4f} ms per (z,M,ell) evaluation")
    
    print("\\nTest 3: Parameter variations (production grid)")
    print("-" * 50)
    
    variations = [
        {"H0": 70.0, "name": "H0=70"},
        {"omega_cdm": 0.13, "name": "Ωcdm=0.13"},
        {"c500": 2.5, "name": "c500=2.5"},
        {"P0GNFW": 8.0, "name": "P0=8.0"},
    ]
    
    param_times = []
    for var in variations:
        test_params = cosmo_params.copy()
        test_params.update({k: v for k, v in var.items() if k != 'name'})
        
        start = time.time()
        ell_var, C_yy_var = tsz_calc.compute_clyy_fast(params_values_dict=test_params)
        elapsed = time.time() - start
        param_times.append(elapsed)
        
        print(f"  {var['name']}: {elapsed:.3f}s")
    
    avg_param_time = np.mean(param_times)
    print(f"\\n✅ Parameter variations: {avg_param_time:.3f}s ± {np.std(param_times):.3f}s")
    
    print("\\nTest 4: Different ell ranges")
    print("-" * 50)
    
    ell_configs = [
        {"lmax": 3000, "dlogell": 0.1, "name": "ell≤3000"},
        {"lmax": 5000, "dlogell": 0.1, "name": "ell≤5000"}, 
        {"lmax": 10000, "dlogell": 0.05, "name": "ell≤10k (fine)"},
    ]
    
    for config in ell_configs:
        start = time.time()
        ell_test, C_yy_test = tsz_calc.compute_clyy_fast(
            cosmo_params, lmax=config["lmax"], dlogell=config["dlogell"]
        )
        elapsed = time.time() - start
        
        if elapsed > 10:  # First call includes compilation
            print(f"  {config['name']}: {elapsed:.3f}s (includes compilation for new grid)")
        else:
            print(f"  {config['name']}: {elapsed:.3f}s")
        print(f"    → {len(ell_test)} ell points")
    
    print("\\n" + "=" * 60)
    print("PRODUCTION GRID PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"Grid Resolution: 100 z × 100 m × {n_ell} ell")
    print(f"Total Evaluations: {total_evals:,}")
    print(f"Compiled Performance: {avg_compiled:.3f}s ± {std_compiled:.3f}s")
    
    if avg_compiled < 3.0:
        print("✅ EXCELLENT: Production grid runs in <3s")
        print("🚀 Perfect for cosmological parameter estimation")
    elif avg_compiled < 10.0:
        print("✅ GOOD: Acceptable for production use")
    else:
        print("⚠️  SLOW: Consider optimization or smaller grids")
    
    print(f"\\nPerformance Metrics:")
    print(f"  • Rate: {rate_ms:.4f} ms per evaluation")
    print(f"  • Throughput: {1/avg_compiled:.1f} power spectra per second")
    print(f"  • MCMC ready: {avg_param_time:.3f}s per parameter step")
    
    # Comparison with different grid sizes
    print(f"\\nGrid Size Comparison (estimated):")
    small_grid_evals = 50 * 50 * n_ell
    large_grid_evals = 150 * 150 * n_ell
    
    small_est = avg_compiled * (small_grid_evals / total_evals)
    large_est = avg_compiled * (large_grid_evals / total_evals)
    
    print(f"  • 50×50 grid: ~{small_est:.3f}s (estimated)")
    print(f"  • 100×100 grid: {avg_compiled:.3f}s (measured)")
    print(f"  • 150×150 grid: ~{large_est:.3f}s (estimated)")
    
    if avg_compiled < 5.0:
        print(f"\\n🎉 100×100 PRODUCTION GRID APPROVED!")
        print("Ready for high-precision cosmological analysis")
    
    print("=" * 60)

if __name__ == "__main__":
    main()