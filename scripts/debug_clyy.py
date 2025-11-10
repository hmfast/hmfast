#!/usr/bin/env python3
"""
Debug tSZ power spectrum calculation - simplified test.
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
    print("Debug tSZ Power Spectrum Calculation")
    print("=" * 60)
    
    tsz_calc = TSZPowerSpectrum()
    
    # Simple cosmological parameters
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
    
    print("Testing individual components...")
    
    # Test 1: Check comoving volume
    z_test = 1.0
    vol = tsz_calc.dVdzdOmega(z_test, cosmo_params)
    print(f"\\n1. Comoving volume at z={z_test}: {vol:.2e} (Mpc/h)³/sr")
    
    # Test 2: Check HMF values
    m_test = jnp.array([1e14, 5e14, 1e15])
    hmf_vals = tsz_calc.emulator.get_hmf_at_z_and_m(z_test, m_test, cosmo_params)
    print(f"\\n2. HMF values at z={z_test}:")
    for i, (m, hmf) in enumerate(zip(m_test, hmf_vals)):
        print(f"   M = {m:.0e} M_sun/h: dndlnM = {hmf:.2e} h³/Mpc³")
    
    # Test 3: Check GNFW pressure profile
    x_test = jnp.array([0.1, 1.0, 5.0])
    m_single = 1e14
    Pe_vals = tsz_calc.gnfw_pressure_profile(x_test, z_test, m_single, cosmo_params)
    print(f"\\n3. GNFW pressure profile at z={z_test}, M={m_single:.0e}:")
    for x, Pe in zip(x_test, Pe_vals):
        print(f"   x = {x}: Pe = {Pe:.2e} eV/cm³")
    
    # Test 4: Check y_ell prefactor
    prefactor = tsz_calc.y_ell_prefactor(z_test, m_test, params_values_dict=cosmo_params)
    print(f"\\n4. y_ell prefactors at z={z_test}:")
    for i, (m, pref) in enumerate(zip(m_test, prefactor)):
        print(f"   M = {m:.0e}: prefactor = {pref:.2e}")
    
    # Test 5: Simple y_ell calculation
    print(f"\\n5. Testing y_ell calculation...")
    try:
        ell_native, y_ell_native = tsz_calc.y_ell_complete(z_test, m_test[:1], cosmo_params)
        print(f"   ✓ y_ell shape: {y_ell_native.shape}")
        print(f"   ✓ ell range: [{ell_native[0].min():.1f}, {ell_native[0].max():.1f}]")
        print(f"   ✓ y_ell range: [{y_ell_native.min():.2e}, {y_ell_native.max():.2e}]")
    except Exception as e:
        print(f"   ✗ y_ell calculation failed: {e}")
        
    # Test 6: Very simple C_l calculation
    print(f"\\n6. Simple C_l calculation (minimal grid)...")
    try:
        start_time = time.time()
        ell, C_yy = tsz_calc.compute_clyy(
            params_values_dict=cosmo_params,
            lmin=100, lmax=300, dlogell=0.3,  # Just 3-4 ell points
            z_min=0.5, z_max=1.5, n_z=10,     # Small z range
            M_min=5e13, M_max=5e14, n_m=10    # Small M range  
        )
        elapsed = time.time() - start_time
        
        print(f"   ✓ Computed {len(ell)} ell points in {elapsed:.3f}s")
        print(f"   ✓ C_l range: [{C_yy.min():.2e}, {C_yy.max():.2e}]")
        
        # Check for typical tSZ values (should be ~1e-12 to 1e-10 in dimensionless units)
        if C_yy.max() > 1e-15:
            print("   ✓ C_l values are in reasonable range")
        else:
            print("   ⚠ C_l values seem very small - possible unit issue")
            
        # Print individual values
        print("   ell values and C_l:")
        for l, cl in zip(ell, C_yy):
            print(f"     ell={l:.1f}: C_l = {cl:.2e}")
            
    except Exception as e:
        print(f"   ✗ C_l calculation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\\n" + "=" * 60)
    print("Debug Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()