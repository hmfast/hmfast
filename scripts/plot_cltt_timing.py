#!/usr/bin/env python3
"""
Standalone cl^TT plotting script with timing analysis.

This script demonstrates the unambiguous cl^TT plotting method for hmfast
and benchmarks performance across different cosmological parameters.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

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

# Set up matplotlib
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
})


def generate_cosmological_parameters(n_params: int = 10) -> List[Dict[str, float]]:
    """
    Generate a set of diverse cosmological parameters for timing tests.
    
    Parameters
    ----------
    n_params : int
        Number of parameter sets to generate
        
    Returns
    -------
    List[Dict[str, float]]
        List of parameter dictionaries
    """
    params_list = []
    
    # Base parameters
    base = {
        'omega_b': 0.02242,
        'omega_cdm': 0.11933,
        'H0': 67.66,
        'tau_reio': 0.0561,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665,
        'log10z_c': 3.562,
        'thetai_scf': 2.83,
    }
    
    # Generate variations
    for i in range(n_params):
        params = base.copy()
        
        # Vary key parameters across their allowed ranges
        params['fEDE'] = 0.001 + (i / (n_params - 1)) * 0.12  # 0.001 to 0.121
        params['H0'] = 65.0 + (i / (n_params - 1)) * 10.0     # 65 to 75
        params['omega_cdm'] = 0.10 + (i / (n_params - 1)) * 0.05  # 0.10 to 0.15
        params['n_s'] = 0.94 + (i / (n_params - 1)) * 0.04    # 0.94 to 0.98
        params['ln10^{10}A_s'] = 2.9 + (i / (n_params - 1)) * 0.2  # 2.9 to 3.1
        
        # Add slight variations to other parameters
        params['omega_b'] = base['omega_b'] * (0.95 + 0.1 * np.random.random())
        params['tau_reio'] = base['tau_reio'] * (0.8 + 0.4 * np.random.random())
        
        params_list.append(params)
    
    return params_list


def time_cltt_calculation(emulator: EDEEmulator, 
                         params_list: List[Dict[str, float]], 
                         lmax: int = 2500) -> Dict[str, Any]:
    """
    Time cl^TT calculations for multiple parameter sets.
    
    Parameters
    ----------
    emulator : EDEEmulator
        Initialized emulator
    params_list : List[Dict[str, float]]
        List of parameter sets
    lmax : int
        Maximum multipole
        
    Returns
    -------
    Dict[str, Any]
        Timing results and spectra
    """
    n_params = len(params_list)
    times = []
    spectra_list = []
    
    print(f"Timing cl^TT calculation for {n_params} parameter sets...")
    
    # JIT compilation run
    print("JIT compiling...")
    _ = emulator.get_cmb_spectra(params_list[0], lmax=lmax)
    print("✓ JIT compilation complete")
    
    # Time each parameter set
    for i, params in enumerate(params_list):
        print(f"  Computing set {i+1}/{n_params} (fEDE={params['fEDE']:.3f})...", end=' ')
        
        start_time = time.time()
        cmb_spectra = emulator.get_cmb_spectra(params, lmax=lmax)
        end_time = time.time()
        
        computation_time = end_time - start_time
        times.append(computation_time)
        spectra_list.append(cmb_spectra)
        
        print(f"{computation_time:.4f}s")
    
    # Statistics
    times = np.array(times)
    results = {
        'times': times,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times),
        'spectra_list': spectra_list,
        'params_list': params_list
    }
    
    print(f"\nTiming Results:")
    print(f"  Mean time: {results['mean_time']:.4f} ± {results['std_time']:.4f} s")
    print(f"  Min time:  {results['min_time']:.4f} s")
    print(f"  Max time:  {results['max_time']:.4f} s")
    print(f"  Total time: {results['total_time']:.3f} s")
    
    return results


def plot_cltt_spectra(timing_results: Dict[str, Any], save_dir: str = 'plots'):
    """
    Plot cl^TT spectra for all parameter sets.
    
    Parameters
    ----------
    timing_results : Dict[str, Any]
        Results from timing calculation
    save_dir : str
        Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    spectra_list = timing_results['spectra_list']
    params_list = timing_results['params_list']
    times = timing_results['times']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color map for different parameter sets
    colors = plt.cm.viridis(np.linspace(0, 1, len(spectra_list)))
    
    # Plot 1: All cl^TT spectra
    for i, (spectra, params, color) in enumerate(zip(spectra_list, params_list, colors)):
        ell = spectra['ell']
        # UNAMBIGUOUS: Direct plotting of D_ℓ values - NO factors needed!
        ax1.plot(ell, spectra['tt'], color=color, linewidth=1.5, alpha=0.8,
                label=f"fEDE={params['fEDE']:.3f}")
    
    ax1.set_xlabel('Multipole ℓ')
    ax1.set_ylabel('ℓ(ℓ+1)C_ℓ^TT/(2π) [μK²]')
    ax1.set_title('cl^TT Spectra for Different Cosmologies')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(2, 2500)
    
    # Plot 2: Relative differences to first spectrum
    reference_spectrum = spectra_list[0]['tt']
    for i, (spectra, params, color) in enumerate(zip(spectra_list[1:], params_list[1:], colors[1:]), 1):
        ell = spectra['ell']
        rel_diff = 100 * (spectra['tt'] - reference_spectrum) / reference_spectrum
        ax2.plot(ell, rel_diff, color=color, linewidth=1.5, alpha=0.8,
                label=f"fEDE={params['fEDE']:.3f}")
    
    ax2.axhline(0, color='k', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Multipole ℓ')
    ax2.set_ylabel('Relative Difference [%]')
    ax2.set_title(f'Relative to Reference (fEDE={params_list[0]["fEDE"]:.3f})')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(2, 2500)
    
    # Plot 3: Timing results
    fede_values = [p['fEDE'] for p in params_list]
    bars = ax3.bar(range(len(times)), times * 1000, color=colors, alpha=0.7)
    ax3.set_xlabel('Parameter Set')
    ax3.set_ylabel('Computation Time [ms]')
    ax3.set_title(f'cl^TT Timing (Mean: {timing_results["mean_time"]*1000:.1f}ms)')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(range(len(times)))
    ax3.set_xticklabels([f'{f:.3f}' for f in fede_values], rotation=45)
    
    # Add timing values on bars
    for bar, time_ms in zip(bars, times * 1000):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time_ms:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Parameter space coverage
    h0_values = [p['H0'] for p in params_list]
    scatter = ax4.scatter(fede_values, h0_values, c=times*1000, cmap='plasma', s=80, alpha=0.8)
    ax4.set_xlabel('fEDE')
    ax4.set_ylabel('H0 [km/s/Mpc]')
    ax4.set_title('Parameter Space Coverage')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Computation Time [ms]')
    
    # Add legends (only for first few entries to avoid clutter)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cltt_timing_analysis.png')
    plt.close()
    
    # Create a simpler comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot only a few representative spectra for clarity
    indices_to_plot = [0, len(spectra_list)//4, len(spectra_list)//2, 3*len(spectra_list)//4, -1]
    colors_simple = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, color in zip(indices_to_plot, colors_simple):
        spectra = spectra_list[i]
        params = params_list[i]
        ell = spectra['ell']
        
        ax.plot(ell, spectra['tt'], color=color, linewidth=2,
               label=f"fEDE={params['fEDE']:.3f}, H0={params['H0']:.1f}")
    
    ax.set_xlabel('Multipole ℓ')
    ax.set_ylabel('ℓ(ℓ+1)C_ℓ^TT/(2π) [μK²]')
    ax.set_title('cl^TT Spectra - Representative Sample')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2, 2500)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cltt_representative_spectra.png')
    plt.close()
    
    print(f"✓ Plots saved to {os.path.abspath(save_dir)}/")


def main():
    """Main function."""
    print("cl^TT Standalone Plotting and Timing Script")
    print("=" * 50)
    
    # Initialize emulator
    print("Initializing EDE emulator...")
    try:
        emulator = EDEEmulator()
        print("✓ Emulator initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize emulator: {e}")
        sys.exit(1)
    
    # Generate parameter sets
    n_param_sets = 10
    print(f"\nGenerating {n_param_sets} cosmological parameter sets...")
    params_list = generate_cosmological_parameters(n_param_sets)
    print("✓ Parameter sets generated")
    
    # Print parameter ranges
    fede_range = [p['fEDE'] for p in params_list]
    h0_range = [p['H0'] for p in params_list]
    print(f"  fEDE range: {min(fede_range):.3f} to {max(fede_range):.3f}")
    print(f"  H0 range: {min(h0_range):.1f} to {max(h0_range):.1f}")
    
    # Time cl^TT calculations
    timing_results = time_cltt_calculation(emulator, params_list, lmax=2500)
    
    # Create plots
    print(f"\nCreating plots...")
    plot_cltt_spectra(timing_results)
    
    # Summary
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Computed cl^TT for {n_param_sets} parameter sets")
    print(f"⚡ Average computation time: {timing_results['mean_time']*1000:.1f} ms")
    print(f"📁 Results saved to plots/")
    
    # Performance assessment
    if timing_results['mean_time'] < 0.1:
        perf = "Excellent"
    elif timing_results['mean_time'] < 0.5:
        perf = "Good"
    else:
        perf = "Acceptable"
    
    print(f"🚀 Performance assessment: {perf} ({timing_results['mean_time']*1000:.1f}ms average)")


if __name__ == "__main__":
    main()