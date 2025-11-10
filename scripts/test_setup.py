#!/usr/bin/env python3
"""
Quick test script to verify EDE-v2 emulator setup.

This script performs basic tests to ensure the emulator is working correctly
before running the full plotting suite.
"""

import os
import sys
import time

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import jax
        print(f"✓ JAX version: {jax.__version__}")
        
        import jax.numpy as jnp
        print("✓ JAX NumPy imported")
        
        import matplotlib
        print(f"✓ Matplotlib version: {matplotlib.__version__}")
        
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_hmfast_import():
    """Test that hmfast can be imported."""
    print("\nTesting hmfast import...")
    
    try:
        from hmfast import EDEEmulator
        print("✓ hmfast.EDEEmulator imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import hmfast: {e}")
        print("Make sure to run: pip install -e . from the hmfast directory")
        return False

def test_data_path():
    """Test that data path is set and exists."""
    print("\nTesting data path...")
    
    data_path = os.getenv('PATH_TO_CLASS_SZ_DATA')
    if data_path is None:
        print("✗ PATH_TO_CLASS_SZ_DATA environment variable not set")
        return False
    
    print(f"✓ PATH_TO_CLASS_SZ_DATA: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"✗ Data directory does not exist: {data_path}")
        return False
    
    print("✓ Data directory exists")
    
    ede_path = os.path.join(data_path, 'ede-v2')
    if not os.path.exists(ede_path):
        print(f"✗ EDE-v2 directory not found: {ede_path}")
        return False
    
    print("✓ EDE-v2 directory exists")
    return True

def test_emulator_initialization():
    """Test that the emulator can be initialized."""
    print("\nTesting emulator initialization...")
    
    try:
        from hmfast import EDEEmulator
        emulator = EDEEmulator()
        print("✓ EDEEmulator initialized successfully")
        return emulator
    except Exception as e:
        print(f"✗ Failed to initialize emulator: {e}")
        return None

def test_basic_calculation(emulator):
    """Test basic emulator calculations."""
    print("\nTesting basic calculations...")
    
    if emulator is None:
        print("✗ No emulator to test")
        return False
    
    # Define test parameters
    test_params = {
        'omega_b': 0.02242,
        'omega_cdm': 0.11933,
        'H0': 67.66,
        'tau_reio': 0.0561,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665,
        'fEDE': 0.05,
        'log10z_c': 3.562,
        'thetai_scf': 2.83,
    }
    
    try:
        # Test parameter validation
        is_valid = emulator.validate_parameters(test_params)
        print(f"✓ Parameter validation: {'Valid' if is_valid else 'Invalid'}")
        
        # Test angular distance
        start_time = time.time()
        da = emulator.get_angular_distance_at_z(1.0, test_params)
        calc_time = time.time() - start_time
        print(f"✓ Angular distance at z=1: {da:.2f} Mpc ({calc_time:.4f}s)")
        
        # Test Hubble parameter
        hz = emulator.get_hubble_at_z(1.0, test_params)
        print(f"✓ Hubble parameter at z=1: {hz:.2f} km/s/Mpc")
        
        # Test power spectrum
        pk, k = emulator.get_pkl_at_z(0.0, test_params)
        print(f"✓ Power spectrum: {len(k)} k-modes")
        
        # Test derived parameters
        derived = emulator.get_derived_parameters(test_params)
        print(f"✓ Derived parameters: σ₈ = {derived.get('sigma8', 'N/A'):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Calculation failed: {e}")
        return False

def test_jax_functionality(emulator):
    """Test JAX-specific functionality."""
    print("\nTesting JAX functionality...")
    
    if emulator is None:
        print("✗ No emulator to test")
        return False
    
    try:
        import jax
        import jax.numpy as jnp
        
        test_params = {
            'omega_b': 0.02242,
            'omega_cdm': 0.11933,
            'H0': 67.66,
            'tau_reio': 0.0561,
            'ln10^{10}A_s': 3.047,
            'n_s': 0.9665,
            'fEDE': 0.05,
            'log10z_c': 3.562,
            'thetai_scf': 2.83,
        }
        
        # Test vectorization
        z_array = jnp.array([0.5, 1.0, 1.5])
        da_array = emulator.get_angular_distance_at_z(z_array, test_params)
        print(f"✓ Vectorized calculation: {len(da_array)} results")
        
        # Test JIT compilation
        @jax.jit
        def jit_test(z_val):
            return emulator.get_angular_distance_at_z(z_val, test_params)
        
        # First call (compilation)
        start_time = time.time()
        result1 = jit_test(1.0)
        compile_time = time.time() - start_time
        
        # Second call (compiled)
        start_time = time.time()
        result2 = jit_test(1.0)
        exec_time = time.time() - start_time
        
        print(f"✓ JIT compilation: {compile_time:.4f}s compile, {exec_time:.6f}s exec")
        print(f"✓ JIT speedup: {compile_time/exec_time:.1f}x")
        
        # Test gradient
        def da_h0(h0_val):
            params = test_params.copy()
            params['H0'] = h0_val
            return emulator.get_angular_distance_at_z(1.0, params)
        
        grad_func = jax.grad(da_h0)
        gradient = grad_func(67.66)
        print(f"✓ Gradient calculation: dDA/dH₀ = {gradient:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ JAX test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("EDE-v2 Emulator Setup Test")
    print("=" * 30)
    
    tests_passed = 0
    total_tests = 5
    
    # Run tests
    if test_imports():
        tests_passed += 1
    
    if test_hmfast_import():
        tests_passed += 1
    
    if test_data_path():
        tests_passed += 1
    
    emulator = test_emulator_initialization()
    if emulator is not None:
        tests_passed += 1
    
    if test_basic_calculation(emulator):
        tests_passed += 1
    
    # Bonus test (doesn't count toward total)
    test_jax_functionality(emulator)
    
    # Summary
    print("\n" + "=" * 30)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Ready to run plotting scripts.")
        print("Run: ./run_ede_plots.sh to generate plots")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. Set PATH_TO_CLASS_SZ_DATA: export PATH_TO_CLASS_SZ_DATA=/path/to/data")
        print("2. Install hmfast: pip install -e .")
        print("3. Install dependencies: pip install jax matplotlib numpy")
        return 1

if __name__ == "__main__":
    sys.exit(main())