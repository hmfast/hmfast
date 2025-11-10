#!/usr/bin/env python3
"""
Simple script to test dVdzdOmega calculation at z=0.
"""

import os
import sys
import jax
import jax.numpy as jnp

# JAX configuration
jax.config.update("jax_enable_x64", True)

# Set up environment
if 'PATH_TO_CLASS_SZ_DATA' not in os.environ:
    os.environ['PATH_TO_CLASS_SZ_DATA'] = '/Users/boris/class_sz_data_directory'

# Add src to path
sys.path.append('../src')
from hmfast.tsz_power import TSZPowerSpectrum

def main():
    print("=" * 50)
    print("Testing dVdzdOmega calculation")
    print("=" * 50)
    
    # Initialize TSZ calculator
    tsz_calc = TSZPowerSpectrum()
    print("✓ TSZPowerSpectrum initialized")
    
    # Test parameters (Planck 2018-like)
    params = {
        'omega_b': 0.02242,
        'omega_cdm': 0.11933,
        'H0': 67.66,
        'tau_reio': 0.0561,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665,
    }
    
    print(f"\nCosmological parameters:")
    print(f"  H0 = {params['H0']} km/s/Mpc")
    print(f"  Omega_b = {params['omega_b']}")
    print(f"  Omega_cdm = {params['omega_cdm']}")
    
    # Get relevant params to show h
    rparams = tsz_calc.emulator.get_all_relevant_params(params)
    h = rparams['h']
    Omega_m = rparams['Omega0_m']
    
    print(f"  h = {h:.4f}")
    print(f"  Omega_m = {Omega_m:.4f}")
    
    # Test dV/dz/dΩ at z=0
    z = 0.0
    dV_dz_dOmega = tsz_calc.dVdzdOmega(z, params_values_dict=params)
    
    print(f"\nComoving volume element at z = {z}:")
    print(f"  dV/dz/dΩ = {dV_dz_dOmega:.6e} (Mpc/h)³ sr⁻¹")
    
    # Test at a few other redshifts for comparison
    print(f"\nComparison at different redshifts:")
    test_redshifts = [0.0, 0.5, 1.0, 2.0, 3.0]
    
    for z_test in test_redshifts:
        dV = tsz_calc.dVdzdOmega(z_test, params_values_dict=params)
        print(f"  z = {z_test:.1f}: dV/dz/dΩ = {dV:.6e} (Mpc/h)³ sr⁻¹")
    
    # Show the formula components at z=0
    print(f"\nFormula breakdown at z = 0:")
    print(f"  dV/dz/dΩ = (1+z)² × d_A(z)² / H(z)")
    
    z = 0.0
    dAz = tsz_calc.emulator.get_angular_distance_at_z(z, params_values_dict=params) * h
    Hz = tsz_calc.emulator.get_hubble_at_z(z, params_values_dict=params) / h
    
    print(f"  (1+z)² = {(1+z)**2:.3f}")
    print(f"  d_A(z) = {dAz:.3f} Mpc/h")  
    print(f"  H(z) = {Hz:.6f} (Mpc/h)⁻¹")
    print(f"  → dV/dz/dΩ = {(1+z)**2 * dAz**2 / Hz:.6e} (Mpc/h)³ sr⁻¹")
    
    print("\n" + "=" * 50)
    print("✅ dVdzdOmega calculation complete!")

if __name__ == "__main__":
    main()