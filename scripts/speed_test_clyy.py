#!/usr/bin/env python3
"""
Speed test for optimized tSZ power spectrum calculation.
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
    print("tSZ Power Spectrum - Speed Optimization Test")
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
    
    print("\\nSpeed Test 1: Fast version (small grid)")
    print("-" * 40)
    
    # Test 1: Fast version
    start_time = time.time()
    try:
        ell_fast, C_yy_fast = tsz_calc.compute_clyy_fast(
            params_values_dict=cosmo_params,
            lmin=10, lmax=10000, dlogell=0.2,  # 16 ell points
            n_z=20, n_m=20  # Small grid
        )
        fast_time = time.time() - start_time
        print(f"Total time (including compilation): {fast_time:.3f}s")
        print(f"Result: {len(ell_fast)} ell points, C_l range: [{C_yy_fast.min():.2e}, {C_yy_fast.max():.2e}]")
        
    except Exception as e:
        print(f"✗ Fast version failed: {e}")
        fast_time = None
    
    print("\\nSpeed Test 2: Second call (compiled)")
    print("-" * 40)
    
    # Test 2: Second call to see compiled speed
    if fast_time is not None:
        start_time = time.time()
        ell_fast2, C_yy_fast2 = tsz_calc.compute_clyy_fast(
            params_values_dict=cosmo_params,
            lmin=10, lmax=10000, dlogell=0.2,
            n_z=20, n_m=20
        )
        compiled_time = time.time() - start_time
        print(f"Compiled speed: {compiled_time:.3f}s")
        print(f"Speedup from compilation: {fast_time/compiled_time:.1f}×")
        
        # Check consistency
        max_diff = float(jnp.max(jnp.abs(C_yy_fast - C_yy_fast2)))
        print(f"Consistency check: max difference = {max_diff:.2e}")
        
    print("\\nSpeed Test 3: Different grid sizes")
    print("-" * 40)
    
    grid_configs = [
        {"name": "Tiny", "n_z": 15, "n_m": 15, "dlogell": 0.3},
        {"name": "Small", "n_z": 20, "n_m": 20, "dlogell": 0.2}, 
        {"name": "Medium", "n_z": 30, "n_m": 30, "dlogell": 0.15},
        {"name": "Large", "n_z": 40, "n_m": 40, "dlogell": 0.1},
    ]
    
    for config in grid_configs:
        print(f"\\n{config['name']} grid ({config['n_z']}×{config['n_m']}):")
        
        start_time = time.time()
        try:
            ell, C_yy = tsz_calc.compute_clyy_fast(
                params_values_dict=cosmo_params,
                lmin=10, lmax=10000, dlogell=config['dlogell'],
                n_z=config['n_z'], n_m=config['n_m']
            )
            elapsed = time.time() - start_time
            
            total_evals = config['n_z'] * config['n_m'] * len(ell)
            rate = elapsed / total_evals * 1000  # ms per evaluation
            
            print(f"  Time: {elapsed:.3f}s for {total_evals:,} evaluations")
            print(f"  Rate: {rate:.3f} ms per (z,M,ell)")
            print(f"  Range: [{C_yy.min():.2e}, {C_yy.max():.2e}]")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\\nSpeed Test 4: Production timing")
    print("-" * 40)
    
    # Test production-quality grid
    print("Production grid (50×50, ell up to 10k):")
    
    start_time = time.time()
    try:
        ell_prod, C_yy_prod = tsz_calc.compute_clyy(
            params_values_dict=cosmo_params,
            lmin=10, lmax=10000, dlogell=0.1,  # 31 ell points  
            n_z=50, n_m=50  # Production quality
        )
        prod_time = time.time() - start_time
        
        print(f"Production time: {prod_time:.3f}s")
        print(f"Grid: {len(ell_prod)} ell × 50 z × 50 masses = {len(ell_prod)*50*50:,} evaluations")
        print(f"Rate: {prod_time/(len(ell_prod)*50*50)*1000:.3f} ms per evaluation")
        
        if prod_time < 1.0:
            print("✅ EXCELLENT: Sub-second production calculation!")
        elif prod_time < 3.0:
            print("✅ GOOD: Fast enough for production use")
        else:
            print("⚠️  SLOW: May need further optimization")
            
    except Exception as e:
        print(f"✗ Production test failed: {e}")
    
    print("\\n" + "=" * 60)
    print("Speed Optimization Summary:")
    print("=" * 60)
    
    print("Key optimizations applied:")
    print("✅ JAX compilation of core functions")
    print("✅ Vectorized HMF and volume calculations") 
    print("✅ Simplified integration (trapezoid vs Simpson)")
    print("✅ Pre-compiled Hankel transforms")
    print("✅ Optimized array operations")
    
    if 'compiled_time' in locals() and compiled_time is not None:
        if compiled_time < 1.0:
            print(f"\\n🚀 TARGET ACHIEVED: {compiled_time:.3f}s < 1s!")
            print("Ready for fast cosmological parameter estimation")
        else:
            print(f"\\n⚠️  Still slower than target: {compiled_time:.3f}s")
            print("Consider further optimizations:")
            print("- Reduce grid sizes")  
            print("- Pre-compute y_ell profiles")
            print("- Use lookup tables for GNFW profiles")
    
    print("=" * 60)

if __name__ == "__main__":
    main()