#!/usr/bin/env python3
"""
EDE-v2 Emulator Plotting Script

This script reproduces key plots from the CLASS-SZ notebooks using the new
JAX-compatible EDE-v2 emulator, demonstrating its capabilities and performance.

IMPORTANT - CMB SPECTRA PLOTTING (NO AMBIGUITY):
The emulator.get_cmb_spectra() method returns D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) values
that are ready for direct plotting. NO additional multiplication by ℓ(ℓ+1)/(2π)
is needed. This is now unambiguous - just plot the returned values directly.

Plots generated:
1. Angular distance vs redshift comparison (EDE vs ΛCDM)
2. Hubble parameter evolution 
3. Linear and nonlinear power spectra
4. sigma8 evolution with redshift
5. CMB power spectra (TT, TE, EE, PP) - DIRECT PLOTTING, NO FACTORS!
6. Derived parameters comparison
7. Performance benchmarks
8. Parameter sensitivity analysis
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set up JAX
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Import hmfast
try:
    from hmfast import EDEEmulator
except ImportError:
    print("Error: Could not import hmfast. Make sure it's installed in the current environment.")
    print("Run: pip install -e . from the hmfast directory")
    sys.exit(1)

# Set up matplotlib for high-quality plots
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

def setup_emulator():
    """Initialize the EDE-v2 emulator with error handling."""
    print("Initializing EDE-v2 emulator...")
    
    try:
        emulator = EDEEmulator()
        print("✓ Emulator initialized successfully")
        return emulator
    except Exception as e:
        print(f"✗ Failed to initialize emulator: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure PATH_TO_CLASS_SZ_DATA is set correctly")
        print("2. Verify that ede-v2/ directory exists in the data path")
        print("3. Check that .npz emulator files are present")
        sys.exit(1)

def define_cosmologies():
    """Define EDE and ΛCDM cosmological parameters."""
    
    # Base parameters
    base_params = {
        'omega_b': 0.02242,
        'omega_cdm': 0.11933,
        'H0': 67.66,
        'tau_reio': 0.0561,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665,
    }
    
    # EDE model parameters
    ede_params = base_params.copy()
    ede_params.update({
        'fEDE': 0.087,         # EDE fraction from Hill et al.
        'log10z_c': 3.562,     # Characteristic redshift
        'thetai_scf': 2.83,    # Scalar field initial condition
    })
    
    # ΛCDM parameters (no EDE)
    lcdm_params = base_params.copy()
    lcdm_params.update({
        'fEDE': 0.0,
        'log10z_c': 3.562,     # Keep same for consistency
        'thetai_scf': 2.83,
    })
    
    return ede_params, lcdm_params

def plot_angular_distances(emulator, ede_params, lcdm_params, save_dir):
    """Plot angular distance evolution (EDE vs ΛCDM)."""
    print("Creating angular distance plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Redshift range
    z_array = jnp.linspace(0.01, 3.0, 100)
    
    # Calculate distances
    start_time = time.time()
    da_ede = emulator.get_angular_distance_at_z(z_array, ede_params)
    da_lcdm = emulator.get_angular_distance_at_z(z_array, lcdm_params)
    calc_time = time.time() - start_time
    
    # Plot absolute distances
    ax1.plot(z_array, da_ede, 'r-', linewidth=2.5, label=f'EDE (f_EDE = {ede_params["fEDE"]:.3f})')
    ax1.plot(z_array, da_lcdm, 'b--', linewidth=2.5, label='ΛCDM')
    
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Angular Distance D_A(z) [Mpc]')
    ax1.set_title('Angular Distance Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)
    
    # Plot relative difference
    rel_diff = 100 * (da_ede - da_lcdm) / da_lcdm
    ax2.plot(z_array, rel_diff, 'g-', linewidth=2.5)
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Relative Difference [%]')
    ax2.set_title('(EDE - ΛCDM) / ΛCDM × 100%')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 3)
    
    # Add computation time annotation
    fig.suptitle(f'Angular Distance Comparison (Computed in {calc_time:.3f}s)', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/angular_distances.png')
    plt.close()

def plot_hubble_evolution(emulator, ede_params, lcdm_params, save_dir):
    """Plot Hubble parameter evolution."""
    print("Creating Hubble parameter plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Redshift range
    z_array = jnp.linspace(0, 5.0, 100)
    
    # Calculate Hubble parameters
    hz_ede = emulator.get_hubble_at_z(z_array, ede_params)
    hz_lcdm = emulator.get_hubble_at_z(z_array, lcdm_params)
    
    # Plot absolute values
    ax1.plot(z_array, hz_ede, 'r-', linewidth=2.5, label='EDE')
    ax1.plot(z_array, hz_lcdm, 'b--', linewidth=2.5, label='ΛCDM')
    
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('H(z) [km/s/Mpc]')
    ax1.set_title('Hubble Parameter Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 5)
    
    # Plot relative difference
    rel_diff = 100 * (hz_ede - hz_lcdm) / hz_lcdm
    ax2.plot(z_array, rel_diff, 'g-', linewidth=2.5)
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Relative Difference [%]')
    ax2.set_title('(EDE - ΛCDM) / ΛCDM × 100%')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/hubble_evolution.png')
    plt.close()

def plot_power_spectra(emulator, ede_params, lcdm_params, save_dir):
    """Plot linear and nonlinear power spectra."""
    print("Creating power spectra plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    redshifts = [0.0, 1.0]
    colors = ['navy', 'darkred']
    
    for i, z in enumerate(redshifts):
        # Get power spectra
        pk_lin_ede, k = emulator.get_pkl_at_z(z, ede_params)
        pk_lin_lcdm, _ = emulator.get_pkl_at_z(z, lcdm_params)
        pk_nl_ede, _ = emulator.get_pknl_at_z(z, ede_params)
        pk_nl_lcdm, _ = emulator.get_pknl_at_z(z, lcdm_params)
        
        # Linear power spectrum
        ax1.loglog(k, pk_lin_ede, color=colors[i], linestyle='-', 
                  linewidth=2, label=f'EDE z={z}')
        ax1.loglog(k, pk_lin_lcdm, color=colors[i], linestyle='--', 
                  linewidth=2, label=f'ΛCDM z={z}')
        
        # Nonlinear power spectrum  
        ax2.loglog(k, pk_nl_ede, color=colors[i], linestyle='-',
                  linewidth=2, label=f'EDE z={z}')
        ax2.loglog(k, pk_nl_lcdm, color=colors[i], linestyle='--',
                  linewidth=2, label=f'ΛCDM z={z}')
        
        # Linear relative difference
        rel_diff_lin = 100 * (pk_lin_ede - pk_lin_lcdm) / pk_lin_lcdm
        ax3.semilogx(k, rel_diff_lin, color=colors[i], linewidth=2, label=f'z={z}')
        
        # Nonlinear relative difference
        rel_diff_nl = 100 * (pk_nl_ede - pk_nl_lcdm) / pk_nl_lcdm
        ax4.semilogx(k, rel_diff_nl, color=colors[i], linewidth=2, label=f'z={z}')
    
    # Format plots
    ax1.set_xlabel('k [h/Mpc]')
    ax1.set_ylabel('P_L(k) [(Mpc/h)³]')
    ax1.set_title('Linear Power Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1e-3, 10)
    
    ax2.set_xlabel('k [h/Mpc]')
    ax2.set_ylabel('P_NL(k) [(Mpc/h)³]')
    ax2.set_title('Nonlinear Power Spectrum')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1e-3, 10)
    
    ax3.set_xlabel('k [h/Mpc]')
    ax3.set_ylabel('(EDE - ΛCDM)/ΛCDM [%]')
    ax3.set_title('Linear P(k) Relative Difference')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle=':', alpha=0.7)
    ax3.set_xlim(1e-3, 10)
    
    ax4.set_xlabel('k [h/Mpc]')
    ax4.set_ylabel('(EDE - ΛCDM)/ΛCDM [%]')
    ax4.set_title('Nonlinear P(k) Relative Difference')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle=':', alpha=0.7)
    ax4.set_xlim(1e-3, 10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/power_spectra.png')
    plt.close()

def plot_sigma8_evolution(emulator, ede_params, lcdm_params, save_dir):
    """Plot sigma8 evolution with redshift."""
    print("Creating sigma8 evolution plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Redshift range
    z_array = jnp.linspace(0, 3.0, 50)
    
    # Calculate sigma8
    s8_ede = emulator.get_sigma8_at_z(z_array, ede_params)
    s8_lcdm = emulator.get_sigma8_at_z(z_array, lcdm_params)
    
    # Plot absolute values
    ax1.plot(z_array, s8_ede, 'r-', linewidth=2.5, label='EDE')
    ax1.plot(z_array, s8_lcdm, 'b--', linewidth=2.5, label='ΛCDM')
    
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('σ₈(z)')
    ax1.set_title('σ₈ Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)
    
    # Plot relative difference
    rel_diff = 100 * (s8_ede - s8_lcdm) / s8_lcdm
    ax2.plot(z_array, rel_diff, 'g-', linewidth=2.5)
    ax2.axhline(y=0, color='k', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Relative Difference [%]')
    ax2.set_title('(EDE - ΛCDM) / ΛCDM × 100%')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 3)
    
    # Add current values annotation
    s8_ede_0 = float(s8_ede[0])
    s8_lcdm_0 = float(s8_lcdm[0])
    ax1.text(0.05, 0.95, f'σ₈(z=0):\nEDE: {s8_ede_0:.4f}\nΛCDM: {s8_lcdm_0:.4f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sigma8_evolution.png')
    plt.close()

def plot_cmb_spectra(emulator, ede_params, lcdm_params, save_dir):
    """Plot CMB power spectra."""
    print("Creating CMB spectra plots...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Get CMB spectra
    start_time = time.time()
    cmb_ede = emulator.get_cmb_spectra(ede_params, lmax=2500)
    cmb_lcdm = emulator.get_cmb_spectra(lcdm_params, lmax=2500)
    end_time = time.time()
    print(f"Time taken to get CMB spectra: {end_time - start_time} seconds")
    ell = cmb_ede['ell']
    
    # TT spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    # Note: cmb_ede['tt'] already contains D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) - NO factor needed!
    ax1.plot(ell, cmb_ede['tt'], 'r-', linewidth=2, label='EDE')
    ax1.plot(ell, cmb_lcdm['tt'], 'b--', linewidth=2, label='ΛCDM')
    ax1.set_xlabel('Multipole ℓ')
    ax1.set_ylabel('ℓ(ℓ+1)C_ℓ^TT/(2π) [μK²]')
    ax1.set_title('Temperature Power Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2, 2500)
    
    # EE spectrum
    ax2 = fig.add_subplot(gs[0, 1])
    # Note: cmb_ede['ee'] already contains D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) - NO factor needed!
    ax2.plot(ell, cmb_ede['ee'] , 'r-', linewidth=2, label='EDE')
    ax2.plot(ell, cmb_lcdm['ee'] , 'b--', linewidth=2, label='ΛCDM')
    ax2.set_xlabel('Multipole ℓ')
    ax2.set_ylabel('ℓ(ℓ+1)C_ℓ^EE/(2π) [μK²]')
    ax2.set_title('E-mode Polarization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2, 2500)
    
    # TE spectrum
    ax3 = fig.add_subplot(gs[1, 0])
    # Note: cmb_ede['te'] already contains D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) - NO factor needed!
    ax3.plot(ell, cmb_ede['te'] , 'r-', linewidth=2, label='EDE')
    ax3.plot(ell, cmb_lcdm['te'] , 'b--', linewidth=2, label='ΛCDM')
    ax3.set_xlabel('Multipole ℓ')
    ax3.set_ylabel('ℓ(ℓ+1)C_ℓ^TE/(2π) [μK²]')
    ax3.set_title('Temperature-E Cross Correlation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(2, 2500)
    
    # PP (lensing) spectrum
    ax4 = fig.add_subplot(gs[1, 1])
    # Note: cmb_ede['pp'] already contains appropriate scaling - NO factor needed!
    ax4.plot(ell[2:], cmb_ede['pp'][2:], 'r-', linewidth=2, label='EDE')
    ax4.plot(ell[2:], cmb_lcdm['pp'][2:], 'b--', linewidth=2, label='ΛCDM')
    ax4.set_xlabel('Multipole ℓ')
    ax4.set_ylabel('ℓ²(ℓ+1)²C_ℓ^φφ/(2π)')
    ax4.set_title('Lensing Potential')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(10, 2500)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cmb_spectra.png')
    plt.close()

def plot_derived_parameters(emulator, ede_params, lcdm_params, save_dir):
    """Plot comparison of derived parameters."""
    print("Creating derived parameters comparison...")
    
    # Get derived parameters
    derived_ede = emulator.get_derived_parameters(ede_params)
    derived_lcdm = emulator.get_derived_parameters(lcdm_params)
    
    # Select key parameters to plot
    key_params = ['sigma8', 'h', 'Omega_m', '100*theta_s']
    values_ede = [float(derived_ede.get(p, 0)) for p in key_params]
    values_lcdm = [float(derived_lcdm.get(p, 0)) for p in key_params]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x_pos = np.arange(len(key_params))
    width = 0.35
    
    # Bar plot
    bars1 = ax1.bar(x_pos - width/2, values_ede, width, label='EDE', color='red', alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, values_lcdm, width, label='ΛCDM', color='blue', alpha=0.7)
    
    ax1.set_xlabel('Parameter')
    ax1.set_ylabel('Value')
    ax1.set_title('Derived Parameters Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(key_params)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar1, bar2, val1, val2 in zip(bars1, bars2, values_ede, values_lcdm):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + height1*0.01,
                f'{val1:.4f}', ha='center', va='bottom', fontsize=10)
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + height2*0.01,
                f'{val2:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Relative differences
    rel_diffs = [100 * (v_ede - v_lcdm) / v_lcdm 
                 for v_ede, v_lcdm in zip(values_ede, values_lcdm)]
    
    colors = ['red' if rd > 0 else 'blue' for rd in rel_diffs]
    bars = ax2.bar(x_pos, rel_diffs, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    
    ax2.set_xlabel('Parameter')
    ax2.set_ylabel('Relative Difference [%]')
    ax2.set_title('(EDE - ΛCDM) / ΛCDM × 100%')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(key_params)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, rel_diffs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{val:.3f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/derived_parameters.png')
    plt.close()

def plot_performance_benchmark(emulator, ede_params, save_dir):
    """Benchmark emulator performance."""
    print("Running performance benchmarks...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Test different numbers of redshifts
    n_redshifts = [1, 5, 10, 50, 100, 500, 1000]
    times_regular = []
    times_jit = []
    
    # Define JIT-compiled function
    @jax.jit
    def jit_angular_distance(z_array):
        return emulator.get_angular_distance_at_z(z_array, ede_params)
    
    for n in n_redshifts:
        z_test = jnp.linspace(0.1, 3.0, n)
        
        # Regular timing
        start_time = time.time()
        _ = emulator.get_angular_distance_at_z(z_test, ede_params)
        times_regular.append(time.time() - start_time)
        
        # JIT timing (after compilation)
        _ = jit_angular_distance(z_test)  # Compilation run
        start_time = time.time()
        _ = jit_angular_distance(z_test)
        times_jit.append(time.time() - start_time)
    
    # Plot 1: Scaling with number of redshifts
    ax1.loglog(n_redshifts, times_regular, 'o-', linewidth=2, markersize=6, label='Regular')
    ax1.loglog(n_redshifts, times_jit, 's-', linewidth=2, markersize=6, label='JIT compiled')
    ax1.set_xlabel('Number of redshifts')
    ax1.set_ylabel('Computation time [s]')
    ax1.set_title('Performance Scaling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup factor
    speedup = np.array(times_regular) / np.array(times_jit)
    ax2.semilogx(n_redshifts, speedup, 'o-', linewidth=2, markersize=6, color='green')
    ax2.set_xlabel('Number of redshifts')
    ax2.set_ylabel('Speedup factor')
    ax2.set_title('JIT Speedup vs Regular')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Method comparison timing
    methods = ['Angular Distance', 'Hubble Parameter', 'Critical Density', 
               'Linear P(k)', 'Nonlinear P(k)', 'sigma8']
    method_times = []
    
    z_test = jnp.linspace(0.1, 3.0, 100)
    
    # Time each method
    start_time = time.time()
    _ = emulator.get_angular_distance_at_z(z_test, ede_params)
    method_times.append(time.time() - start_time)
    
    start_time = time.time()
    _ = emulator.get_hubble_at_z(z_test, ede_params)
    method_times.append(time.time() - start_time)
    
    start_time = time.time()
    _ = emulator.get_rho_crit_at_z(z_test, ede_params)
    method_times.append(time.time() - start_time)
    
    start_time = time.time()
    _ = emulator.get_pkl_at_z(1.0, ede_params)
    method_times.append(time.time() - start_time)
    
    start_time = time.time()
    _ = emulator.get_pknl_at_z(1.0, ede_params)
    method_times.append(time.time() - start_time)
    
    start_time = time.time()
    _ = emulator.get_sigma8_at_z(z_test, ede_params)
    method_times.append(time.time() - start_time)
    
    bars = ax3.bar(methods, method_times, color='skyblue', alpha=0.7)
    ax3.set_ylabel('Computation time [s]')
    ax3.set_title('Method Performance (100 redshifts)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add time labels on bars
    for bar, time_val in zip(bars, method_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.4f}s', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Memory usage comparison (approximate)
    memory_items = ['Single redshift', '100 redshifts', '1000 redshifts', 'CMB spectra']
    # Approximate memory usage in MB (these are estimates)
    memory_usage = [0.001, 0.01, 0.1, 5.0]
    
    ax4.bar(memory_items, memory_usage, color='lightcoral', alpha=0.7)
    ax4.set_ylabel('Approximate Memory [MB]')
    ax4.set_title('Memory Usage Estimates')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_benchmark.png')
    plt.close()

def plot_parameter_sensitivity(emulator, ede_params, save_dir):
    """Plot parameter sensitivity analysis."""
    print("Creating parameter sensitivity plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Test different EDE fractions
    fede_values = jnp.linspace(0.0, 0.15, 8)
    colors = plt.cm.viridis(np.linspace(0, 1, len(fede_values)))
    
    z_test = jnp.linspace(0.1, 3.0, 50)
    
    for i, fede in enumerate(fede_values):
        test_params = ede_params.copy()
        test_params['fEDE'] = float(fede)
        
        # Angular distance sensitivity
        da = emulator.get_angular_distance_at_z(z_test, test_params)
        ax1.plot(z_test, da, color=colors[i], linewidth=2, 
                label=f'fEDE = {fede:.3f}')
    
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('D_A(z) [Mpc]')
    ax1.set_title('fEDE Sensitivity - Angular Distance')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Hubble parameter sensitivity
    for i, fede in enumerate(fede_values):
        test_params = ede_params.copy()
        test_params['fEDE'] = float(fede)
        
        hz = emulator.get_hubble_at_z(z_test, test_params)
        ax2.plot(z_test, hz, color=colors[i], linewidth=2,
                label=f'fEDE = {fede:.3f}')
    
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('H(z) [km/s/Mpc]')
    ax2.set_title('fEDE Sensitivity - Hubble Parameter')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Test different log10z_c values
    log10zc_values = jnp.linspace(3.0, 4.0, 6)
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(log10zc_values)))
    
    for i, log10zc in enumerate(log10zc_values):
        test_params = ede_params.copy()
        test_params['log10z_c'] = float(log10zc)
        
        # sigma8 sensitivity
        s8 = emulator.get_sigma8_at_z(z_test, test_params)
        ax3.plot(z_test, s8, color=colors2[i], linewidth=2,
                label=f'log₁₀z_c = {log10zc:.2f}')
    
    ax3.set_xlabel('Redshift z')
    ax3.set_ylabel('σ₈(z)')
    ax3.set_title('log₁₀z_c Sensitivity - σ₈ Evolution')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # H0 sensitivity
    h0_values = jnp.linspace(65, 75, 6)
    colors3 = plt.cm.coolwarm(np.linspace(0, 1, len(h0_values)))
    
    for i, h0 in enumerate(h0_values):
        test_params = ede_params.copy()
        test_params['H0'] = float(h0)
        
        # Angular distance at z=1
        da_z1 = emulator.get_angular_distance_at_z(z_test, test_params)
        ax4.plot(z_test, da_z1, color=colors3[i], linewidth=2,
                label=f'H₀ = {h0:.1f}')
    
    ax4.set_xlabel('Redshift z')
    ax4.set_ylabel('D_A(z) [Mpc]')
    ax4.set_title('H₀ Sensitivity - Angular Distance')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/parameter_sensitivity.png')
    plt.close()

def create_summary_plot(emulator, ede_params, lcdm_params, save_dir):
    """Create a comprehensive summary plot."""
    print("Creating summary plot...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    z_array = jnp.linspace(0.01, 3.0, 100)
    
    # 1. Angular distances
    ax1 = fig.add_subplot(gs[0, 0])
    da_ede = emulator.get_angular_distance_at_z(z_array, ede_params)
    da_lcdm = emulator.get_angular_distance_at_z(z_array, lcdm_params)
    ax1.plot(z_array, da_ede, 'r-', linewidth=2, label='EDE')
    ax1.plot(z_array, da_lcdm, 'b--', linewidth=2, label='ΛCDM')
    ax1.set_xlabel('z')
    ax1.set_ylabel('D_A [Mpc]')
    ax1.set_title('Angular Distance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Hubble parameter
    ax2 = fig.add_subplot(gs[0, 1])
    hz_ede = emulator.get_hubble_at_z(z_array, ede_params)
    hz_lcdm = emulator.get_hubble_at_z(z_array, lcdm_params)
    ax2.plot(z_array, hz_ede, 'r-', linewidth=2, label='EDE')
    ax2.plot(z_array, hz_lcdm, 'b--', linewidth=2, label='ΛCDM')
    ax2.set_xlabel('z')
    ax2.set_ylabel('H(z) [km/s/Mpc]')
    ax2.set_title('Hubble Parameter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. sigma8
    ax3 = fig.add_subplot(gs[0, 2])
    s8_ede = emulator.get_sigma8_at_z(z_array, ede_params)
    s8_lcdm = emulator.get_sigma8_at_z(z_array, lcdm_params)
    ax3.plot(z_array, s8_ede, 'r-', linewidth=2, label='EDE')
    ax3.plot(z_array, s8_lcdm, 'b--', linewidth=2, label='ΛCDM')
    ax3.set_xlabel('z')
    ax3.set_ylabel('σ₈(z)')
    ax3.set_title('σ₈ Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Power spectrum at z=0
    ax4 = fig.add_subplot(gs[0, 3])
    pk_ede, k = emulator.get_pkl_at_z(0.0, ede_params)
    pk_lcdm, _ = emulator.get_pkl_at_z(0.0, lcdm_params)
    ax4.loglog(k, pk_ede, 'r-', linewidth=2, label='EDE')
    ax4.loglog(k, pk_lcdm, 'b--', linewidth=2, label='ΛCDM')
    ax4.set_xlabel('k [h/Mpc]')
    ax4.set_ylabel('P(k) [(Mpc/h)³]')
    ax4.set_title('Linear P(k) at z=0')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5-8. CMB Spectra
    cmb_ede = emulator.get_cmb_spectra(ede_params, lmax=2000)
    cmb_lcdm = emulator.get_cmb_spectra(lcdm_params, lmax=2000)
    ell = cmb_ede['ell']
    fac = ell * (ell + 1) / (2 * np.pi)
    
    # TT
    ax5 = fig.add_subplot(gs[1, 0])
    # IMPORTANT: get_cmb_spectra() returns D_ℓ values - NO multiplication needed!
    # This is now UNAMBIGUOUS: direct plotting without any factors
    ax5.plot(ell, cmb_ede['tt'], 'r-', linewidth=2, label='EDE')
    ax5.plot(ell, cmb_lcdm['tt'], 'b--', linewidth=2, label='ΛCDM')
    ax5.set_xlabel('ℓ')
    ax5.set_ylabel('ℓ(ℓ+1)C_ℓ^TT/(2π)')
    ax5.set_title('CMB TT')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # EE
    ax6 = fig.add_subplot(gs[1, 1])
    # IMPORTANT: get_cmb_spectra() returns D_ℓ values - NO multiplication needed!
    ax6.plot(ell, cmb_ede['ee'] , 'r-', linewidth=2, label='EDE')
    ax6.plot(ell, cmb_lcdm['ee'] , 'b--', linewidth=2, label='ΛCDM')
    ax6.set_xlabel('ℓ')
    ax6.set_ylabel('ℓ(ℓ+1)C_ℓ^EE/(2π)')
    ax6.set_title('CMB EE')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # TE
    ax7 = fig.add_subplot(gs[1, 2])
    # IMPORTANT: get_cmb_spectra() returns D_ℓ values - NO multiplication needed!
    # Removing the *fac multiplication that was causing confusion
    ax7.plot(ell, cmb_ede['te'], 'r-', linewidth=2, label='EDE')
    ax7.plot(ell, cmb_lcdm['te'], 'b--', linewidth=2, label='ΛCDM')
    ax7.set_xlabel('ℓ')
    ax7.set_ylabel('ℓ(ℓ+1)C_ℓ^TE/(2π)')
    ax7.set_title('CMB TE')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # PP
    ax8 = fig.add_subplot(gs[1, 3])
    # IMPORTANT: get_cmb_spectra() returns appropriately scaled values - NO multiplication needed!
    # Removing the *fac_pp multiplication that was causing confusion
    ax8.plot(ell[2:], cmb_ede['pp'][2:], 'r-', linewidth=2, label='EDE')
    ax8.plot(ell[2:], cmb_lcdm['pp'][2:], 'b--', linewidth=2, label='ΛCDM')
    ax8.set_xlabel('ℓ')
    ax8.set_ylabel('ℓ²(ℓ+1)²C_ℓ^φφ/(2π)')
    ax8.set_title('CMB Lensing')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9-12. Relative differences
    ax9 = fig.add_subplot(gs[2, 0])
    rel_da = 100 * (da_ede - da_lcdm) / da_lcdm
    ax9.plot(z_array, rel_da, 'g-', linewidth=2)
    ax9.axhline(0, color='k', linestyle=':', alpha=0.7)
    ax9.set_xlabel('z')
    ax9.set_ylabel('Δ D_A / D_A [%]')
    ax9.set_title('Angular Distance Diff.')
    ax9.grid(True, alpha=0.3)
    
    ax10 = fig.add_subplot(gs[2, 1])
    rel_hz = 100 * (hz_ede - hz_lcdm) / hz_lcdm
    ax10.plot(z_array, rel_hz, 'g-', linewidth=2)
    ax10.axhline(0, color='k', linestyle=':', alpha=0.7)
    ax10.set_xlabel('z')
    ax10.set_ylabel('Δ H(z) / H(z) [%]')
    ax10.set_title('Hubble Difference')
    ax10.grid(True, alpha=0.3)
    
    ax11 = fig.add_subplot(gs[2, 2])
    rel_s8 = 100 * (s8_ede - s8_lcdm) / s8_lcdm
    ax11.plot(z_array, rel_s8, 'g-', linewidth=2)
    ax11.axhline(0, color='k', linestyle=':', alpha=0.7)
    ax11.set_xlabel('z')
    ax11.set_ylabel('Δ σ₈ / σ₈ [%]')
    ax11.set_title('σ₈ Difference')
    ax11.grid(True, alpha=0.3)
    
    ax12 = fig.add_subplot(gs[2, 3])
    rel_pk = 100 * (pk_ede - pk_lcdm) / pk_lcdm
    ax12.semilogx(k, rel_pk, 'g-', linewidth=2)
    ax12.axhline(0, color='k', linestyle=':', alpha=0.7)
    ax12.set_xlabel('k [h/Mpc]')
    ax12.set_ylabel('Δ P(k) / P(k) [%]')
    ax12.set_title('Power Spectrum Diff.')
    ax12.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(f'EDE-v2 Emulator Summary (fEDE = {ede_params["fEDE"]:.3f})', fontsize=20)
    
    plt.savefig(f'{save_dir}/summary_plot.png')
    plt.close()

def main():
    """Main plotting function."""
    print("EDE-v2 Emulator Plotting Script")
    print("=" * 40)
    
    # Create plots directory
    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving plots to: {os.path.abspath(save_dir)}")
    
    # Initialize emulator
    emulator = setup_emulator()
    
    # Define cosmologies
    ede_params, lcdm_params = define_cosmologies()
    print(f"\nEDE parameters: fEDE = {ede_params['fEDE']:.3f}")
    print(f"ΛCDM parameters: fEDE = {lcdm_params['fEDE']:.3f}")
    
    # Validate parameters
    print("\nValidating parameters...")
    if emulator.validate_parameters(ede_params):
        print("✓ EDE parameters valid")
    else:
        print("⚠ EDE parameters may be outside training range")
    
    if emulator.validate_parameters(lcdm_params):
        print("✓ ΛCDM parameters valid")
    else:
        print("⚠ ΛCDM parameters may be outside training range")
    
    # Generate all plots
    total_start_time = time.time()
    
    plot_angular_distances(emulator, ede_params, lcdm_params, save_dir)
    plot_hubble_evolution(emulator, ede_params, lcdm_params, save_dir)
    plot_power_spectra(emulator, ede_params, lcdm_params, save_dir)
    plot_sigma8_evolution(emulator, ede_params, lcdm_params, save_dir)
    plot_cmb_spectra(emulator, ede_params, lcdm_params, save_dir)
    plot_derived_parameters(emulator, ede_params, lcdm_params, save_dir)
    plot_performance_benchmark(emulator, ede_params, save_dir)
    plot_parameter_sensitivity(emulator, ede_params, save_dir)
    create_summary_plot(emulator, ede_params, lcdm_params, save_dir)
    
    total_time = time.time() - total_start_time
    
    print(f"\n🎉 All plots completed successfully!")
    print(f"📊 Total plotting time: {total_time:.2f} seconds")
    print(f"📁 Plots saved in: {os.path.abspath(save_dir)}")
    
    # List generated files
    plot_files = [f for f in os.listdir(save_dir) if f.endswith(('.png', '.pdf'))]
    print(f"📋 Generated {len(plot_files)} plot files:")
    for file in sorted(plot_files):
        print(f"   • {file}")

if __name__ == "__main__":
    main()