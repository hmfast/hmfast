#!/usr/bin/env python3
"""
Example usage of the EDE-v2 emulator in hmfast.

This example demonstrates how to use the new JAX-compatible EDE emulator
that replaces the classy_szfast dependency while maintaining the same interface.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from hmfast import EDEEmulator

# Enable 64-bit precision for better accuracy
jax.config.update("jax_enable_x64", True)

def main():
    """Main example function."""
    print("EDE-v2 Emulator Example")
    print("=" * 50)
    
    # Check if data path is available
    data_path = os.getenv('PATH_TO_CLASS_SZ_DATA')
    if data_path is None:
        print("Error: PATH_TO_CLASS_SZ_DATA environment variable not set")
        print("Please set it to the path containing the emulator data")
        return
    
    # Initialize the EDE emulator
    print("Initializing EDE-v2 emulator...")
    try:
        emulator = EDEEmulator(data_path=data_path)
        print("✓ Emulator initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize emulator: {e}")
        return
    
    # Define cosmological parameters (EDE-v2 model)
    cosmo_params = {
        'omega_b': 0.02242,
        'omega_cdm': 0.11933,
        'H0': 67.66,
        'tau_reio': 0.0561,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665,
        'fEDE': 0.05,           # Early Dark Energy fraction
        'log10z_c': 3.562,      # log10 of characteristic redshift
        'thetai_scf': 2.83,     # Initial condition for scalar field
    }
    
    print("\nCosmological parameters:")
    for key, value in cosmo_params.items():
        print(f"  {key}: {value}")
    
    # Validate parameters
    if emulator.validate_parameters(cosmo_params):
        print("✓ Parameters are within valid ranges")
    else:
        print("✗ Warning: Some parameters may be outside training ranges")
    
    # Example 1: Angular distance vs redshift
    print("\n1. Computing angular distances...")
    z_array = jnp.linspace(0.1, 3.0, 30)
    
    # Time the computation
    import time
    start_time = time.time()
    da_array = emulator.get_angular_distance_at_z(z_array, cosmo_params)
    end_time = time.time()
    
    print(f"✓ Computed {len(z_array)} angular distances in {end_time - start_time:.4f} seconds")
    print(f"  DA(z=1) = {emulator.get_angular_distance_at_z(1.0, cosmo_params):.2f} Mpc")
    
    # Example 2: Hubble parameter
    print("\n2. Computing Hubble parameters...")
    h_array = emulator.get_hubble_at_z(z_array, cosmo_params)
    print(f"  H(z=0) = {emulator.get_hubble_at_z(0.0, cosmo_params):.2f} km/s/Mpc")
    print(f"  H(z=1) = {emulator.get_hubble_at_z(1.0, cosmo_params):.2f} km/s/Mpc")
    
    # Example 3: Critical density
    print("\n3. Computing critical densities...")
    rho_crit = emulator.get_rho_crit_at_z(1.0, cosmo_params)
    print(f"  ρ_crit(z=1) = {rho_crit:.2e} (Msun/h)/(Mpc/h)^3")
    
    # Example 4: Power spectra
    print("\n4. Computing power spectra...")
    pk_lin, k = emulator.get_pkl_at_z(0.0, cosmo_params)
    pk_nl, _ = emulator.get_pknl_at_z(0.0, cosmo_params)
    
    print(f"✓ Computed power spectra with {len(k)} k-modes")
    print(f"  k range: {k[0]:.4f} - {k[-1]:.2f} h/Mpc")
    
    # Example 5: sigma8 evolution
    print("\n5. Computing sigma8 evolution...")
    z_s8 = jnp.linspace(0., 3., 20)
    s8_array = emulator.get_sigma8_at_z(z_s8, cosmo_params)
    print(f"  σ8(z=0) = {s8_array[0]:.4f}")
    print(f"  σ8(z=1) = {emulator.get_sigma8_at_z(1.0, cosmo_params):.4f}")
    
    # Example 6: Derived parameters
    print("\n6. Computing derived parameters...")
    derived = emulator.get_derived_parameters(cosmo_params)
    print("  Key derived parameters:")
    for param in ['h', 'Omega_m', 'sigma8', '100*theta_s']:
        if param in derived:
            print(f"    {param}: {derived[param]:.4f}")
    
    # Example 7: CMB spectra
    print("\n7. Computing CMB power spectra...")
    cmb_spectra = emulator.get_cmb_spectra(cosmo_params, lmax=2000)
    print(f"✓ Computed CMB spectra up to ℓ_max = {max(cmb_spectra['ell'])}")
    
    # Example 8: JAX compatibility - gradient calculation
    print("\n8. Testing JAX compatibility...")
    
    def angular_distance_h0(h0_val):
        """Function to compute angular distance as function of H0."""
        test_params = cosmo_params.copy()
        test_params['H0'] = h0_val
        return emulator.get_angular_distance_at_z(1.0, test_params)
    
    # Calculate gradient with respect to H0
    grad_func = jax.grad(angular_distance_h0)
    gradient = grad_func(67.66)
    print(f"✓ dDA/dH0 at z=1: {gradient:.4f} Mpc/(km/s/Mpc)")
    
    # Example 9: JIT compilation speed test
    print("\n9. Testing JIT compilation...")
    
    @jax.jit
    def jit_angular_distance(z_val):
        return emulator.get_angular_distance_at_z(z_val, cosmo_params)
    
    # First call (compilation + execution)
    start_time = time.time()
    result1 = jit_angular_distance(1.0)
    compile_time = time.time() - start_time
    
    # Second call (execution only)
    start_time = time.time()
    result2 = jit_angular_distance(1.0)
    exec_time = time.time() - start_time
    
    print(f"✓ First call (with compilation): {compile_time:.4f} seconds")
    print(f"✓ Second call (compiled): {exec_time:.6f} seconds")
    print(f"✓ Speedup factor: {compile_time/exec_time:.1f}x")
    
    # Example 10: Compare with and without EDE
    print("\n10. Comparing EDE vs ΛCDM...")
    
    # ΛCDM parameters (no EDE)
    lcdm_params = cosmo_params.copy()
    lcdm_params['fEDE'] = 0.0
    
    # Compare angular distances
    da_ede = emulator.get_angular_distance_at_z(1.0, cosmo_params)
    da_lcdm = emulator.get_angular_distance_at_z(1.0, lcdm_params)
    
    print(f"  DA(z=1) with EDE (fEDE={cosmo_params['fEDE']}): {da_ede:.2f} Mpc")
    print(f"  DA(z=1) ΛCDM (fEDE=0): {da_lcdm:.2f} Mpc")
    print(f"  Relative difference: {100 * (da_ede - da_lcdm) / da_lcdm:.2f}%")
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("The EDE-v2 emulator is ready for use in your cosmological analyses.")


if __name__ == "__main__":
    main()