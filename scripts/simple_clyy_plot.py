#!/usr/bin/env python3
"""
Simple script to plot one tSZ power spectrum C_l^yy.
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

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
    print("Simple tSZ Power Spectrum Plot")
    print("=" * 50)
    
    # Initialize TSZ calculator
    print("Initializing TSZ calculator...")
    tsz_calc = TSZPowerSpectrum()
    print("✓ TSZPowerSpectrum initialized")
    
    # Cosmological parameters (Planck 2018-like)
    params = {
        'omega_b': 0.02242,
        'omega_cdm': 0.11933,
        'H0': 67.66,
        'tau_reio': 0.0561,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665,
        'B': 1.0,           # Mass bias
        'c500': 1.81,       # Concentration
        'gammaGNFW': 0.31,  # GNFW γ
        'alphaGNFW': 1.33,  # GNFW α  
        'betaGNFW': 4.13,   # GNFW β
        'P0GNFW': 6.41,     # GNFW P0
    }
    
    print("\nComputing C_l^yy...")
    print("Grid: 100z × 100m × ell[10→10000]")
    
    # Compute power spectrum
    start_time = time.time()
    ell, C_yy = tsz_calc.compute_clyy_fast(params)
    elapsed = time.time() - start_time
    
    # Convert to dimensionless units 
    T_CMB_uK = 2.7255e6  # CMB temperature in μK
    C_yy_dimensionless = C_yy * T_CMB_uK**2
    
    print(f"✓ Computed in {elapsed:.3f}s")
    print(f"✓ {len(ell)} multipoles")
    print(f"✓ C_l range: [{C_yy_dimensionless.min():.2e}, {C_yy_dimensionless.max():.2e}] [μK²]")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Convert to D_l = l(l+1)C_l/(2π) for plotting 
    D_l = ell * (ell + 1) * C_yy_dimensionless / (2 * np.pi)
    
    plt.loglog(ell, D_l, 'b-', linewidth=2, label='tSZ 1-halo term')
    
    plt.xlabel('Multipole $\\ell$', fontsize=12)
    plt.ylabel('$D_\\ell^{yy} = \\ell(\\ell+1)C_\\ell^{yy}/(2\\pi)$ [μK²]', fontsize=12)
    plt.title('tSZ Power Spectrum (1-halo term)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some info to the plot
    peak_idx = jnp.argmax(D_l)
    plt.axvline(ell[peak_idx], color='red', linestyle='--', alpha=0.7, 
                label=f'Peak at $\\ell$={ell[peak_idx]:.0f}')
    plt.legend()
    
    # Add computation time as text
    plt.text(0.02, 0.98, f'Computed in {elapsed:.2f}s', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.xlim(10, 10000)
    plt.tight_layout()
    
    # Save plot
    output_file = 'plots/simple_clyy.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    
    # Show key results
    print(f"\n🎯 Key Results:")
    print(f"  Peak at ell = {ell[peak_idx]:.0f}")
    print(f"  Peak D_l = {D_l[peak_idx]:.2e} (dimensionless)")
    print(f"  Total computation: {elapsed:.2f}s")
    print(f"  T_CMB = {T_CMB_uK:.0f} μK (used for normalization)")
    
    print(f"\n✅ Simple tSZ power spectrum plot complete!")

if __name__ == "__main__":
    main()