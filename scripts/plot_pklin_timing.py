#!/usr/bin/env python3
"""
Standalone P(k) linear plotting script with timing analysis.

This script demonstrates the P(k) linear calculation method for hmfast
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


def time_pklin_calculation(emulator: EDEEmulator, 
                          params_list: List[Dict[str, float]], 
                          z: float = 0.0) -> Dict[str, Any]:
    """
    Time P(k) linear calculations for multiple parameter sets.
    
    Parameters
    ----------
    emulator : EDEEmulator
        The emulator instance
    params_list : List[Dict[str, float]]
        List of parameter dictionaries
    z : float
        Redshift for P(k) calculation
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing timing results and spectra
    """
    print(f"Timing P(k) linear calculations at z={z} for {len(params_list)} parameter sets...")
    
    times = []
    pk_spectra = []
    k_array = None
    
    # Warm up JIT compilation with first calculation
    print("Warming up JIT compilation...")
    _, k_array = emulator.calculate_pkl_at_z(z, params_list[0])
    
    # Time each calculation
    for i, params in enumerate(params_list):
        start_time = time.perf_counter()
        pk_spectrum, _ = emulator.calculate_pkl_at_z(z, params)
        end_time = time.perf_counter()
        
        calculation_time = (end_time - start_time) * 1000  # Convert to milliseconds
        times.append(calculation_time)
        pk_spectra.append(pk_spectrum)
        
        print(f"  Parameter set {i+1:2d}: {calculation_time:6.2f} ms (fEDE={params['fEDE']:.3f})")
    
    # Statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nTiming Statistics:")
    print(f"  Mean: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  Range: {min_time:.2f} - {max_time:.2f} ms")
    
    return {
        'times': times,
        'pk_spectra': pk_spectra,
        'k_array': k_array,
        'params_list': params_list,
        'mean_time': mean_time,
        'std_time': std_time,
        'z': z
    }


def plot_pklin_results(results: Dict[str, Any], output_dir: str = 'plots') -> None:
    """
    Create plots showing P(k) linear variations and timing results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from time_pklin_calculation
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pk_spectra = results['pk_spectra']
    k_array = results['k_array']
    params_list = results['params_list']
    times = results['times']
    z = results['z']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Main P(k) plot
    ax1 = plt.subplot(2, 2, (1, 2))
    
    # Color scheme
    colors = plt.cm.viridis(np.linspace(0, 1, len(pk_spectra)))
    
    for i, (pk, params, color) in enumerate(zip(pk_spectra, params_list, colors)):
        # UNAMBIGUOUS: Direct plotting of P(k) values - NO factors needed!
        ax1.loglog(k_array, pk, color=color, linewidth=1.5, alpha=0.8,
                  label=f'fEDE={params["fEDE"]:.3f}')
    
    ax1.set_xlabel('k [h/Mpc]')
    ax1.set_ylabel('P(k) [(Mpc/h)³]')
    ax1.set_title(f'Linear Matter Power Spectrum at z={z}')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # k range annotations for EDE-v2 range (5e-4 to 10 h/Mpc)
    ax1.axvline(x=1e-3, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(1.2e-3, 1e-4, 'Large scales\n(k < 10⁻³)', rotation=90, fontsize=9, alpha=0.7)
    ax1.text(1.2, 1e-4, 'Small scales\n(k > 1)', rotation=90, fontsize=9, alpha=0.7)
    
    # Timing histogram
    ax2 = plt.subplot(2, 2, 3)
    ax2.hist(times, bins=8, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(x=results['mean_time'], color='red', linestyle='--', 
                label=f'Mean: {results["mean_time"]:.2f} ms')
    ax2.set_xlabel('Calculation Time [ms]')
    ax2.set_ylabel('Frequency')
    ax2.set_title('P(k) Calculation Time Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Parameter correlation plot
    ax3 = plt.subplot(2, 2, 4)
    fede_values = [p['fEDE'] for p in params_list]
    ax3.scatter(fede_values, times, c=colors, s=60, alpha=0.8, edgecolors='black')
    ax3.set_xlabel('fEDE')
    ax3.set_ylabel('Calculation Time [ms]')
    ax3.set_title('Time vs fEDE Parameter')
    ax3.grid(True, alpha=0.3)
    
    # Add performance summary text
    performance_text = (
        f"Performance Summary (z={z}):\n"
        f"• Mean time: {results['mean_time']:.1f} ± {results['std_time']:.1f} ms\n"
        f"• Parameter sets: {len(params_list)}\n"
        f"• k range: {k_array[0]:.1e} - {k_array[-1]:.1f} h/Mpc\n"
        f"• EDE-v2 emulator range (classy_szfast compatible)\n"
        f"• Method: calculate_pkl_at_z()"
    )
    
    fig.text(0.02, 0.02, performance_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f'pklin_timing_analysis_z{z:.1f}.png')
    plt.savefig(output_file)
    print(f"\nPlot saved to: {output_file}")
    
    # Also save at multiple redshifts for comparison
    if z == 0.0:
        plt.savefig(os.path.join(output_dir, 'pklin_timing_analysis_z0.png'))
    
    plt.show()


def main():
    """Main function to run P(k) linear timing analysis."""
    print("=" * 60)
    print("hmfast P(k) Linear Timing Analysis")
    print("=" * 60)
    
    # Check environment variable
    if 'PATH_TO_CLASS_SZ_DATA' not in os.environ:
        print("Warning: PATH_TO_CLASS_SZ_DATA environment variable not set.")
        print("Setting to default: /Users/boris/class_sz_data_directory")
        os.environ['PATH_TO_CLASS_SZ_DATA'] = '/Users/boris/class_sz_data_directory'
    
    try:
        # Initialize emulator
        print("Initializing EDEEmulator...")
        emulator = EDEEmulator()
        print("✓ Successfully loaded EDE emulator")
        
        # Generate parameter sets
        print("\nGenerating cosmological parameters...")
        params_list = generate_cosmological_parameters(n_params=10)
        print(f"✓ Generated {len(params_list)} parameter sets")
        
        # Print parameter ranges
        fede_range = [p['fEDE'] for p in params_list]
        h0_range = [p['H0'] for p in params_list]
        print(f"  fEDE range: {min(fede_range):.3f} - {max(fede_range):.3f}")
        print(f"  H0 range: {min(h0_range):.1f} - {max(h0_range):.1f}")
        
        # Test multiple redshifts
        test_redshifts = [0.0, 0.5, 1.0, 2.0]
        all_results = {}
        
        for z in test_redshifts:
            print(f"\n{'='*40}")
            print(f"Testing at redshift z = {z}")
            print(f"{'='*40}")
            results = time_pklin_calculation(emulator, params_list, z=z)
            all_results[z] = results
            
            # Create plots for each redshift
            print(f"\nCreating plots for z={z}...")
            plot_pklin_results(results)
        
        # Summary across all redshifts
        print("\n" + "=" * 60)
        print("P(k) Linear Timing Analysis Complete!")
        print("\nSummary across all redshifts:")
        for z, results in all_results.items():
            print(f"  z={z}: {results['mean_time']:.2f} ± {results['std_time']:.2f} ms")
        
        # Overall average
        all_times = []
        for results in all_results.values():
            all_times.extend(results['times'])
        overall_mean = np.mean(all_times)
        overall_std = np.std(all_times)
        print(f"\nOverall average: {overall_mean:.2f} ± {overall_std:.2f} ms")
        print(f"Total calculations: {len(all_times)}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()