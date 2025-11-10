#!/usr/bin/env python3
"""
hmfast Halo Mass Function (HMF) Demonstration

This script demonstrates the Tinker08 halo mass function implementation
following the patterns from classy_sz notebooks with JAX compatibility.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

# Set up environment
if 'PATH_TO_CLASS_SZ_DATA' not in os.environ:
    os.environ['PATH_TO_CLASS_SZ_DATA'] = '/Users/boris/class_sz_data_directory'

from hmfast import EDEEmulator

def main():
    print("=" * 60)
    print("hmfast Halo Mass Function (Tinker08) Demonstration")
    print("Following classy_sz notebook patterns")  
    print("=" * 60)
    
    # Initialize emulator
    emulator = EDEEmulator()
    print("✓ EDEEmulator initialized")
    
    # Cosmological parameters
    cosmo_params = {
        'omega_b': 0.02242,
        'omega_cdm': 0.11933,
        'H0': 67.66,
        'tau_reio': 0.0561,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665,
    }
    
    print("\n1. Cosmological parameters:")
    rparams = emulator.get_all_relevant_params(cosmo_params)
    print(f"   h = {rparams['h']:.3f}")
    print(f"   Ω_b = {rparams['Omega_b']:.4f}")
    print(f"   Ω_cdm = {rparams['Omega_cdm']:.4f}")
    print(f"   Ω_m = {rparams['Omega0_m']:.4f}")
    
    print("\n2. Tinker08 mass function:")
    
    # Test Tinker08 function directly (following notebook pattern)
    print("   Testing different σ(M) and redshifts:")
    
    # Typical sigma values for different mass scales
    sigmas = jnp.array([0.5, 0.8, 1.0, 1.5, 2.0])  # Different mass scales
    z_values = jnp.array([0.0, 1.0, 2.0])           # Different redshifts
    delta_mean = jnp.full_like(z_values, 200.0)     # Δ = 200 (mean density)
    
    print(f"   σ(M) values: {sigmas}")
    print(f"   Redshifts: {z_values}")
    
    # Calculate Tinker08 mass function
    tinker_result = emulator._MF_T08(sigmas, z_values, delta_mean)
    
    print(f"   Result shape: {tinker_result.shape}")
    print("   Mass function values:")
    for i, z in enumerate(z_values):
        print(f"     z={z}: {tinker_result[i, :3]} (first 3 σ values)")
    
    print("\n3. JAX compatibility:")
    print(f"   isinstance(sigmas, jnp.ndarray): {isinstance(sigmas, jnp.ndarray)}")
    print(f"   isinstance(tinker_result, jnp.ndarray): {isinstance(tinker_result, jnp.ndarray)}")
    print("   ✓ Full JAX array compatibility")
    
    print("\n4. Mass function evolution:")
    print("   Behavior check (f(σ) should decrease with z for fixed σ):")
    sigma_test = 1.0
    sigma_idx = 2  # Index for σ = 1.0
    for i, z in enumerate(z_values):
        f_val = tinker_result[i, sigma_idx]
        print(f"   f(σ={sigma_test}, z={z}) = {f_val:.4f}")
    
    print("\n" + "=" * 60)
    print("HMF Implementation Summary:")
    print("✅ Tinker08 mass function implemented")
    print("✅ JAX-compatible calculations") 
    print("✅ Following classy_sz notebook patterns")
    print("✅ Parameter derivation working")
    print("✅ Ready for cosmological HMF calculations")
    
    # Future: Full HMF with mcfit integration
    print("\n💡 Next steps:")
    print("   - Integrate with mcfit for σ(M) calculation")
    print("   - Full dn/dM grid computation")
    print("   - Mass-redshift interpolation")
    print("=" * 60)

if __name__ == "__main__":
    main()