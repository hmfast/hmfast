#!/usr/bin/env python3
"""
hmfast Halo Mass Function Plotting Script

This script creates plots of the Tinker08 halo mass function
following the style and patterns from classy_sz notebooks.
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

# Set up matplotlib (following classy_sz notebook style)
plt.style.use('default')
plt.rcParams.update({
    'font.size': 20,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.labelsize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def main():
    print("=" * 60)
    print("hmfast Halo Mass Function Plotting")
    print("=" * 60)
    
    # Initialize emulator
    emulator = EDEEmulator()
    
    # Cosmological parameters
    cosmo_params = {
        'omega_b': 0.02242,
        'omega_cdm': 0.11933,
        'H0': 67.66,
        'tau_reio': 0.0561,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665,
    }
    
    rparams = emulator.get_all_relevant_params(cosmo_params)
    
    # Create figure (following classy_sz notebook layout)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    
    # Simulate mass function for different σ values and redshifts
    # (This demonstrates the concept - full implementation would use mcfit)
    
    # Panel 1: Mass function f(σ) vs σ for different z
    ax = ax1
    ax.tick_params(axis='x', which='both', length=5, direction='in', pad=10)
    ax.tick_params(axis='y', which='both', length=5, direction='in', pad=5)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.grid(visible=True, which="both", alpha=0.1, linestyle='--')
    
    # σ range (represents different mass scales)
    sigma_range = jnp.logspace(-0.5, 1.0, 100)  # 0.3 to 10
    
    # Calculate for different redshifts
    z_values = jnp.array([0.0, 1.0, 2.0])
    colors = ['k', 'k', 'k']
    linestyles = ['-', '--', '-.']
    
    for i, (z, color, ls) in enumerate(zip(z_values, colors, linestyles)):
        delta_mean = jnp.full_like(jnp.array([z]), 200.0)
        z_array = jnp.array([z])
        
        # Calculate Tinker08 mass function
        f_sigma = emulator._MF_T08(sigma_range, z_array, delta_mean)[0, :]
        
        ax.plot(sigma_range, f_sigma, label=f'z={z:.0f}', alpha=1., 
                color=color, linestyle=ls, linewidth=2)
    
    ax.set_xlabel(r'$\sigma(M)$')
    ax.set_ylabel(r'$f(\sigma)$')
    ax.set_title('Tinker08 Mass Function')
    ax.set_xscale('log')
    ax.legend(loc='best', frameon=False)
    
    # Panel 2: Evolution with overdensity Δ
    ax = ax2
    ax.tick_params(axis='x', which='both', length=5, direction='in', pad=10)
    ax.tick_params(axis='y', which='both', length=5, direction='in', pad=5)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.grid(visible=True, which="both", alpha=0.1, linestyle='--')
    
    # Different overdensity values
    delta_values = [200, 400, 800]
    z_fixed = 0.0
    sigma_test = jnp.logspace(-0.3, 0.7, 50)
    
    for i, delta in enumerate(delta_values):
        delta_mean = jnp.array([float(delta)])
        z_array = jnp.array([z_fixed])
        
        f_sigma = emulator._MF_T08(sigma_test, z_array, delta_mean)[0, :]
        ax.plot(sigma_test, f_sigma, label=f'Δ={delta}', alpha=1., 
                color='k', linestyle=linestyles[i], linewidth=2)
    
    ax.set_xlabel(r'$\sigma(M)$')
    ax.set_ylabel(r'$f(\sigma)$')
    ax.set_title(f'Overdensity Dependence (z={z_fixed})')
    ax.set_xscale('log')
    ax.legend(loc='best', frameon=False)
    
    # Panel 3: Redshift evolution for fixed σ
    ax = ax3
    ax.tick_params(axis='x', which='both', length=5, direction='in', pad=10)
    ax.tick_params(axis='y', which='both', length=5, direction='in', pad=5)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.grid(visible=True, which="both", alpha=0.1, linestyle='--')
    
    z_range = jnp.linspace(0, 3, 50)
    sigma_values = [0.5, 1.0, 1.5]
    delta_mean = jnp.full_like(z_range, 200.0)
    
    for i, sigma in enumerate(sigma_values):
        sigma_array = jnp.full_like(z_range, sigma)
        f_values = []
        
        for z in z_range:
            f_val = emulator._MF_T08(jnp.array([sigma]), jnp.array([z]), jnp.array([200.0]))[0, 0]
            f_values.append(f_val)
        
        ax.plot(z_range, f_values, label=f'σ={sigma}', alpha=1., 
                color='k', linestyle=linestyles[i], linewidth=2)
    
    ax.set_xlabel('Redshift z')
    ax.set_ylabel(r'$f(\sigma)$')
    ax.set_title('Redshift Evolution')
    ax.legend(loc='best', frameon=False)
    
    # Panel 4: Parameter information
    ax = ax4
    ax.axis('off')
    
    # Display cosmological parameters
    param_text = (
        f"Cosmological Parameters:\\n"
        f"h = {rparams['h']:.3f}\\n"
        f"Ω_b = {rparams['Omega_b']:.4f}\\n"
        f"Ω_cdm = {rparams['Omega_cdm']:.4f}\\n"
        f"Ω_m = {rparams['Omega0_m']:.4f}\\n"
        f"\\nTinker08 Mass Function\\n"
        f"Overdensity: Δ = 200 (mean)\\n"
        f"Implementation: JAX-compatible\\n"
        f"Following classy_sz patterns"
    )
    
    ax.text(0.1, 0.9, param_text, transform=ax.transAxes, fontsize=16,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    output_file = 'plots/hmf_tinker08_demo.png'
    plt.savefig(output_file)
    print(f"✓ Plot saved to: {output_file}")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("HMF Plotting Complete!")
    print("✅ Tinker08 mass function visualized")
    print("✅ Following classy_sz notebook style") 
    print("✅ JAX-compatible implementation")
    print("=" * 60)

if __name__ == "__main__":
    main()