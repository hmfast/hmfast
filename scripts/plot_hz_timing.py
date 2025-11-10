#!/usr/bin/env python3
"""
Hubble Parameter H(z) plotting script with timing analysis.

This script demonstrates the H(z) calculation method for hmfast
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
    
    # Base parameters from test file
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
        params['omega_b'] = base['omega_b'] * (0.95 + 0.1 * np.random.random())
        
        params_list.append(params)
    
    return params_list


def time_hz_calculation(emulator: EDEEmulator, 
                        params_list: List[Dict[str, float]], 
                        z_test: float = 1.0) -> Dict[str, Any]:
    """
    Time H(z) calculations for multiple parameter sets.
    
    Parameters
    ----------
    emulator : EDEEmulator
        The emulator instance
    params_list : List[Dict[str, float]]
        List of parameter dictionaries
    z_test : float
        Test redshift for timing
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing timing results and H(z) values
    """
    print(f"Timing H(z) calculations at z={z_test} for {len(params_list)} parameter sets...")
    print(f"Following classy_szfast.py implementation pattern")
    print()
    
    times = []
    hz_values_mpc = []
    hz_values_kms = []
    
    # Warm up JIT compilation with first calculation
    print("Warming up JIT compilation...")
    _ = emulator.get_hubble_at_z(z_test, params_list[0])
    
    # Time each calculation
    for i, params in enumerate(params_list):
        start_time = time.perf_counter()
        hz_mpc = emulator.get_hubble_at_z(z_test, params, units="1/Mpc")
        hz_kms = emulator.get_hubble_at_z(z_test, params, units="km/s/Mpc")
        end_time = time.perf_counter()
        
        calculation_time = (end_time - start_time) * 1000  # Convert to milliseconds
        times.append(calculation_time)
        hz_values_mpc.append(float(hz_mpc))
        hz_values_kms.append(float(hz_kms))
        
        print(f"  Parameter set {i+1:2d}: {calculation_time:6.2f} ms (H0={params['H0']:.1f}) -> H({z_test})={hz_kms:.1f} km/s/Mpc")
    
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
        'hz_values_mpc': hz_values_mpc,
        'hz_values_kms': hz_values_kms,
        'params_list': params_list,
        'mean_time': mean_time,
        'std_time': std_time,
        'z_test': z_test
    }


def plot_hz_results(results: Dict[str, Any], output_dir: str = 'plots') -> None:
    """
    Create plots showing H(z) variations and timing results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results from time_hz_calculation
    output_dir : str
        Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    hz_values_kms = results['hz_values_kms']
    params_list = results['params_list']
    times = results['times']
    z_test = results['z_test']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Main H(z) redshift evolution plot
    ax1 = plt.subplot(2, 2, (1, 2))
    
    # Plot H(z) evolution across redshifts for first few parameter sets
    z_range = np.linspace(0.0, 20.0, 500)
    colors = plt.cm.viridis(np.linspace(0, 1, min(5, len(params_list))))
    
    emulator = EDEEmulator()
    
    for i, (params, color) in enumerate(zip(params_list[:5], colors)):
        try:
            hz_evolution = emulator.get_hubble_at_z(z_range, params, units="km/s/Mpc")
            # Only plot valid (non-NaN) values
            valid_mask = ~np.isnan(hz_evolution)
            if valid_mask.any():
                ax1.plot(z_range[valid_mask], hz_evolution[valid_mask], 
                        color=color, linewidth=2, alpha=0.8,
                        label=f'H0={params["H0"]:.1f}, fEDE={params["fEDE"]:.3f}')
        except Exception as e:
            print(f"Warning: Could not plot evolution for parameter set {i+1}: {e}")
    
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('H(z) [km/s/Mpc]')
    ax1.set_title('Hubble Parameter Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 20)
    ax1.set_yscale('log')
    
    # Mark test redshift
    ax1.axvline(x=z_test, color='red', linestyle='--', alpha=0.7, linewidth=1)
    ax1.text(z_test + 0.1, ax1.get_ylim()[1]*0.9, f'Test z={z_test}', 
             rotation=90, fontsize=9, alpha=0.7)
    
    # Timing histogram
    ax2 = plt.subplot(2, 2, 3)
    ax2.hist(times, bins=8, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(x=results['mean_time'], color='red', linestyle='--', 
                label=f'Mean: {results["mean_time"]:.2f} ms')
    ax2.set_xlabel('Calculation Time [ms]')
    ax2.set_ylabel('Frequency')
    ax2.set_title('H(z) Calculation Time Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Parameter correlation plot
    ax3 = plt.subplot(2, 2, 4)
    h0_values = [p['H0'] for p in params_list]
    ax3.scatter(h0_values, hz_values_kms, c=colors[:len(params_list)] if len(colors) >= len(params_list) else 'blue', 
                s=60, alpha=0.8, edgecolors='black')
    ax3.set_xlabel('H0 [km/s/Mpc]')
    ax3.set_ylabel(f'H(z={z_test}) [km/s/Mpc]')
    ax3.set_title('H(z) vs H0 Parameter')
    ax3.grid(True, alpha=0.3)
    
    # Add performance summary text
    performance_text = (
        f"Performance Summary (z={z_test}):\n"
        f"• Mean time: {results['mean_time']:.1f} ± {results['std_time']:.1f} ms\n"
        f"• Parameter sets: {len(params_list)}\n"
        f"• Valid for all z ≥ 0\n"
        f"• Method: get_hubble_at_z()\n"
        f"✓ Following classy_szfast patterns"
    )
    
    fig.text(0.02, 0.02, performance_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f'hz_timing_analysis_z{z_test:.1f}.png')
    plt.savefig(output_file)
    print(f"\nPlot saved to: {output_file}")
    
    plt.show()


def main():
    """Main function to run H(z) timing analysis."""
    print("=" * 60)
    print("hmfast Hubble Parameter H(z) Timing Analysis")
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
        
        # Time calculations at z=1
        results = time_hz_calculation(emulator, params_list, z_test=1.0)
        
        # Create plots
        print("\nCreating plots...")
        plot_hz_results(results)
        
        print("\n" + "=" * 60)
        print("H(z) Timing Analysis Complete!")
        print(f"Average calculation time: {results['mean_time']:.2f} ms")
        print(f"H(z) implementation following classy_szfast patterns")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()