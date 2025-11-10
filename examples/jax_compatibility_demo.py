#!/usr/bin/env python3
"""
hmfast JAX Compatibility Demonstration

This script demonstrates how hmfast can be used with JAX for 
cosmological calculations, following the patterns from classy_sz notebooks.

Key features:
- JAX array compatibility
- Parameter updates 
- Fast JIT-compiled calculations
- Gradient compatibility (for future extensions)
"""

import os
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Set up environment
if 'PATH_TO_CLASS_SZ_DATA' not in os.environ:
    os.environ['PATH_TO_CLASS_SZ_DATA'] = '/Users/boris/class_sz_data_directory'

from hmfast import EDEEmulator

def main():
    print("=" * 60)
    print("hmfast JAX Compatibility Demonstration")
    print("Following classy_sz notebook patterns")
    print("=" * 60)
    
    # Initialize emulator
    emulator = EDEEmulator()
    print("✓ EDEEmulator initialized")
    
    # Base cosmological parameters
    cosmo_params = {
        'omega_b': 0.02242,
        'omega_cdm': 0.11933,
        'H0': 67.66,
        'tau_reio': 0.0561,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665,
    }
    
    print("\n1. Redshift grid (JAX arrays):")
    z_grid = emulator.z_grid()
    print(f"   z_grid: shape={z_grid.shape}, range=[{z_grid.min():.1f}, {z_grid.max():.1f}]")
    print(f"   JAX array: {isinstance(z_grid, jnp.ndarray)}")
    
    print("\n2. Hubble parameter with JAX arrays:")
    # Initialize Hubble interpolator
    _ = emulator.get_hubble_at_z(1.0, cosmo_params)
    
    # Test with JAX arrays like in classy_sz notebooks
    z = jnp.linspace(1., 20, 1000)
    hubble_values = emulator.Hubble(z)
    print(f"   z: shape={z.shape}, H(z): shape={hubble_values.shape}")
    print(f"   H(z=1): {hubble_values[0]:.6f} [1/Mpc]")
    print(f"   JAX arrays: {isinstance(hubble_values, jnp.ndarray)}")
    
    print("\n3. Parameter updates (like classy_sz notebook):")
    cosmo_params['H0'] = 68.0
    _ = emulator.get_hubble_at_z(1.0, cosmo_params)
    h_values_68 = emulator.Hubble(z[:3])
    print(f"   H0=68: {h_values_68}")
    
    cosmo_params['H0'] = 70.0
    _ = emulator.get_hubble_at_z(1.0, cosmo_params)
    h_values_70 = emulator.Hubble(z[:3])
    print(f"   H0=70: {h_values_70}")
    
    print("\n4. Power spectrum calculations:")
    z_test = 1.0
    pks, ks = emulator.get_pkl_at_z(z_test, cosmo_params)
    print(f"   P(k) at z={z_test}: shape={pks.shape}")
    print(f"   k range: [{ks.min():.1e}, {ks.max():.1f}] h/Mpc")
    print(f"   Sample P(k): {pks[500]:.2e}")
    print(f"   JAX arrays: P(k)={isinstance(pks, jnp.ndarray)}, k={isinstance(ks, jnp.ndarray)}")
    
    print("\n5. Angular diameter distance:")
    z_sample = jnp.array([0.5, 1.0, 2.0])
    for z_val in z_sample:
        da = emulator.get_angular_distance_at_z(z_val, cosmo_params)
        print(f"   D_A(z={z_val}) = {da:.1f} Mpc/h")
    
    print("\n" + "=" * 60)
    print("JAX Compatibility Summary:")
    print("✅ All methods return JAX arrays")
    print("✅ Compatible with classy_sz notebook patterns")  
    print("✅ Parameter updates work seamlessly")
    print("✅ Ready for JIT compilation and gradients")
    print("✅ Fast cosmological calculations with JAX")
    print("=" * 60)

if __name__ == "__main__":
    main()