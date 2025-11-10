#!/usr/bin/env python3
"""
Plot and time tSZ power spectrum C_l^yy calculation.

This script demonstrates the tSZ power spectrum calculation using the 1-halo term
following the tszsbi pattern but implemented in hmfast.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
jax.config.update("jax_enable_x64", True)

# Set up environment
if 'PATH_TO_CLASS_SZ_DATA' not in os.environ:
    os.environ['PATH_TO_CLASS_SZ_DATA'] = '/Users/boris/class_sz_data_directory'

# Import hmfast TSZ power spectrum
import sys
sys.path.append('../src')
from hmfast.tsz_power import TSZPowerSpectrum

def main():
    print("=" * 60)
    print("tSZ Power Spectrum C_l^yy - Plot and Timing")  
    print("=" * 60)
    
    # Initialize TSZ calculator
    tsz_calc = TSZPowerSpectrum()
    print("✓ TSZPowerSpectrum initialized")
    
    # Cosmological parameters (matching tszsbi defaults)
    cosmo_params = {
        'omega_b': 0.02242,
        'omega_cdm': 0.11933, 
        'H0': 67.66,
        'tau_reio': 0.0561,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665,
        # GNFW parameters
        'B': 1.0,           # Mass calibration
        'c500': 1.81,       # Concentration  
        'gammaGNFW': 0.31,  # GNFW gamma
        'alphaGNFW': 1.33,  # GNFW alpha
        'betaGNFW': 4.13,   # GNFW beta
        'P0GNFW': 6.41,     # GNFW P0
    }
    
    print("✓ Cosmological parameters set")
    
    # Test different ell grid specifications
    print("\\nTesting different ell grid specifications:")
    
    test_configs = [
        {"name": "Coarse", "lmin": 10, "lmax": 10000, "dlogell": 0.2, "n_z": 30, "n_m": 30},
        {"name": "Medium", "lmin": 10, "lmax": 10000, "dlogell": 0.1, "n_z": 50, "n_m": 50},  
        {"name": "Fine", "lmin": 10, "lmax": 10000, "dlogell": 0.05, "n_z": 70, "n_m": 70},
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\\n{i+1}. {config['name']} resolution:")
        print(f"   ell: [{config['lmin']}, {config['lmax']}], Δlog(ell) = {config['dlogell']}")
        print(f"   Integration: {config['n_z']} z × {config['n_m']} masses")
        
        start_time = time.time()
        
        try:
            ell, C_yy = tsz_calc.compute_clyy(
                params_values_dict=cosmo_params,
                lmin=config['lmin'],
                lmax=config['lmax'], 
                dlogell=config['dlogell'],
                z_min=0.01,
                z_max=3.0,
                M_min=1e13,
                M_max=1e16,
                n_z=config['n_z'],
                n_m=config['n_m']
            )
            
            elapsed = time.time() - start_time
            
            # Validate results
            n_ell = len(ell)
            C_yy_range = [float(C_yy.min()), float(C_yy.max())]
            
            print(f"   ✓ Computed in {elapsed:.3f}s")
            print(f"   ✓ Grid: {n_ell} ell points")
            print(f"   ✓ C_l range: [{C_yy_range[0]:.2e}, {C_yy_range[1]:.2e}] μK²")
            
            results.append({
                'name': config['name'],
                'ell': ell,
                'C_yy': C_yy,
                'time': elapsed,
                'n_ell': n_ell,
                'config': config
            })
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            continue
    
    if not results:
        print("\\n✗ No successful calculations")
        return
        
    # Create plots
    print("\\nCreating plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: C_l^yy power spectrum
    ax1.set_title('tSZ Power Spectrum $C_\\ell^{yy}$', fontsize=14)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, result in enumerate(results):
        ell = result['ell']
        C_yy = result['C_yy']
        
        # Convert to μK² (assuming result is in suitable units)
        C_yy_muK2 = C_yy * 1e12  # Adjust scaling as needed
        
        ax1.loglog(ell, ell*(ell+1)/(2*jnp.pi) * C_yy_muK2, 
                  label=f"{result['name']}: {result['n_ell']} points", 
                  color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Multipole $\\ell$')
    ax1.set_ylabel('$\\ell(\\ell+1)C_\\ell^{yy}/(2\\pi)$ [μK²]')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(10, 10000)
    
    # Plot 2: Timing comparison
    ax2.set_title('Computation Time vs Resolution', fontsize=14)
    
    n_ell_vals = [r['n_ell'] for r in results]
    times = [r['time'] for r in results]
    names = [r['name'] for r in results]
    
    bars = ax2.bar(names, times, color=colors[:len(results)], alpha=0.7)
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Computation Time by Resolution')
    
    # Add time labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax2.annotate(f'{time_val:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Plot 3: Scaling analysis
    ax3.set_title('Performance Scaling', fontsize=14)
    
    grid_sizes = [r['config']['n_z'] * r['config']['n_m'] for r in results]
    ax3.loglog(grid_sizes, times, 'o-', linewidth=2, markersize=8, color='red')
    
    # Add reference scaling lines
    grid_ref = jnp.array(grid_sizes)
    linear_ref = times[0] * (grid_ref / grid_sizes[0])
    quadratic_ref = times[0] * (grid_ref / grid_sizes[0])**2
    
    ax3.loglog(grid_ref, linear_ref, '--', alpha=0.5, color='gray', label='Linear scaling')
    ax3.loglog(grid_ref, quadratic_ref, '--', alpha=0.5, color='blue', label='Quadratic scaling')
    
    ax3.set_xlabel('Grid size (n_z × n_m)')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter space info
    ax4.axis('off')
    ax4.set_title('Calculation Parameters', fontsize=14)
    
    param_text = (
        f"Cosmological Parameters:\\n"
        f"H₀ = {cosmo_params['H0']:.1f} km/s/Mpc\\n"
        f"Ωb = {cosmo_params['omega_b']:.4f}\\n" 
        f"Ωcdm = {cosmo_params['omega_cdm']:.4f}\\n"
        f"ns = {cosmo_params['n_s']:.3f}\\n\\n"
        f"GNFW Profile Parameters:\\n"
        f"c₅₀₀ = {cosmo_params['c500']:.2f}\\n"
        f"γ = {cosmo_params['gammaGNFW']:.2f}\\n"
        f"α = {cosmo_params['alphaGNFW']:.2f}\\n" 
        f"β = {cosmo_params['betaGNFW']:.2f}\\n"
        f"P₀ = {cosmo_params['P0GNFW']:.2f}\\n\\n"
        f"Integration Ranges:\\n"
        f"z: [0.01, 3.0]\\n"
        f"M: [10¹³, 10¹⁶] M☉/h\\n"
        f"1-halo term only"
    )
    
    ax4.text(0.1, 0.9, param_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    output_file = 'plots/clyy_tsz_power_spectrum.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    plt.show()
    
    # Summary statistics
    print("\\n" + "=" * 60)
    print("Performance Summary:")
    print("=" * 60)
    
    for result in results:
        config = result['config']
        print(f"{result['name']} resolution:")
        print(f"  Grid: {config['n_z']} × {config['n_m']} = {config['n_z']*config['n_m']} points")
        print(f"  Multipoles: {result['n_ell']} ell values")
        print(f"  Time: {result['time']:.3f} seconds")
        print(f"  Rate: {result['time']/(config['n_z']*config['n_m']*result['n_ell'])*1000:.2f} ms per (z,M,ell)")
        print(f"  C_l range: [{float(result['C_yy'].min()):.2e}, {float(result['C_yy'].max()):.2e}]")
        print()
        
    print("✅ tSZ power spectrum calculation complete!")
    print("✅ 1-halo term implementation following tszsbi pattern") 
    print("✅ Full GNFW profile with Hankel transforms")
    print("✅ JAX-accelerated with hmfast HMF integration")
    print("=" * 60)

if __name__ == "__main__":
    main()