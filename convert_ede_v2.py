#!/usr/bin/env python3
"""
Convert EDE-v2 TensorFlow-based .npz emulator files to pure numpy format.

This script converts the original TensorFlow-based emulator files from the 'ede' 
directory to pure numpy format suitable for JAX in the 'ede_v2_numpy' directory.

The conversion extracts TensorFlow tensors and converts them to numpy arrays,
making the emulators usable without TensorFlow dependencies.
"""

import os
import numpy as np
import sys

def convert_tensorflow_objects(obj):
    """
    Recursively convert TensorFlow objects to numpy arrays.
    
    Parameters
    ----------
    obj : any
        Object that might contain TensorFlow tensors
        
    Returns
    -------
    any
        Converted object with numpy arrays instead of TensorFlow tensors
    """
    if hasattr(obj, 'numpy'):
        # TensorFlow tensor - convert to numpy
        return obj.numpy()
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        # Iterable (list, tuple, etc.) - recursively convert elements
        if hasattr(obj, '_storage'):
            # TensorFlow ListWrapper or similar
            return [convert_tensorflow_objects(item) for item in obj]
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_tensorflow_objects(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: convert_tensorflow_objects(value) for key, value in obj.items()}
    
    # Return as-is for numpy arrays, scalars, etc.
    return obj

def convert_npz_file(input_path, output_path):
    """
    Convert a single .npz emulator file from TensorFlow to numpy format.
    
    Parameters
    ----------
    input_path : str
        Path to input .npz file with TensorFlow objects
    output_path : str
        Path for output .npz file with pure numpy arrays
    """
    print(f"Converting: {os.path.basename(input_path)}")
    
    try:
        # Load the original file
        original_data = np.load(input_path, allow_pickle=True)
            
        # Extract the main data (usually stored as arr_0)
        if 'arr_0' in original_data:
            emulator_data = original_data['arr_0'].flatten()[0]
        else:
            print(f"  Warning: No 'arr_0' found in {input_path}")
            original_data.close()
            return False
            
        # Convert TensorFlow objects to numpy
        print(f"  - Converting {len(emulator_data)} keys...")
        converted_data = convert_tensorflow_objects(emulator_data)
        
        # Verify conversion worked
        if not isinstance(converted_data, dict):
            print(f"  Error: Conversion failed, result is not dict")
            return False
            
        # Count successful conversions
        tf_to_numpy_count = 0
        numpy_preserved_count = 0
        
        for key, value in converted_data.items():
            if isinstance(value, np.ndarray):
                numpy_preserved_count += 1
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                tf_to_numpy_count += 1
        
        print(f"  - Converted {tf_to_numpy_count} TF objects, preserved {numpy_preserved_count} numpy arrays")
        
        # Save as pure numpy
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, converted_data)
        
        # Close the original file
        original_data.close()
        
        # Verify the saved file can be loaded
        test_load = np.load(output_path, allow_pickle=True)
        if 'arr_0' in test_load:
            print(f"  ✓ Saved successfully with {len(test_load['arr_0'].flatten()[0])} keys")
            test_load.close()
            return True
        else:
            print(f"  ✗ Failed to save properly")
            test_load.close()
            return False
            
    except Exception as e:
        print(f"  ✗ Error converting {input_path}: {e}")
        return False

def convert_ede_v2_emulators(input_dir, output_dir):
    """
    Convert all EDE-v2 emulator files from TensorFlow to numpy format.
    
    Parameters
    ----------
    input_dir : str
        Directory containing original TensorFlow-based emulator files
    output_dir : str  
        Directory for converted numpy-based emulator files
    """
    print("EDE-v2 Emulator Conversion: TensorFlow → Numpy")
    print("=" * 50)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return False
    
    # Define the v2 files we need (as used by EDEEmulator)
    file_mapping = {
        # CMB Temperature, E-mode, TE cross-correlation
        'TTTEEE/TT_v2.npz': 'TTTEEE/TT_v2.npz',
        'TTTEEE/EE_v2.npz': 'TTTEEE/EE_v2.npz', 
        'TTTEEE/TE_v2.npz': 'TTTEEE/TE_v2.npz',
        
        # Lensing potential
        'PP/PP_v2.npz': 'PP/PP_v2.npz',
        
        # Linear and nonlinear power spectra
        'PK/PKL_v2.npz': 'PK/PKL_v2.npz',
        'PK/PKNL_v2.npz': 'PK/PKNL_v2.npz',
        
        # Derived parameters
        'derived-parameters/DER_v2.npz': 'derived-parameters/DER_v2.npz',
        
        # Growth and distances
        'growth-and-distances/DAZ_v2.npz': 'growth-and-distances/DAZ_v2.npz',
        'growth-and-distances/HZ_v2.npz': 'growth-and-distances/HZ_v2.npz',
        'growth-and-distances/S8Z_v2.npz': 'growth-and-distances/S8Z_v2.npz',
    }
    
    converted_count = 0
    failed_count = 0
    
    for rel_input_path, rel_output_path in file_mapping.items():
        input_path = os.path.join(input_dir, rel_input_path)
        output_path = os.path.join(output_dir, rel_output_path)
        
        if os.path.exists(input_path):
            success = convert_npz_file(input_path, output_path)
            if success:
                converted_count += 1
            else:
                failed_count += 1
        else:
            print(f"Warning: Input file not found: {input_path}")
            failed_count += 1
    
    print()
    print("=" * 50)
    if failed_count == 0:
        print(f"✓ Conversion successful! Converted {converted_count}/{len(file_mapping)} files")
        print(f"✓ Output directory: {output_dir}")
        print()
        print("The converted files can now be used with EDEEmulator without TensorFlow.")
        return True
    else:
        print(f"⚠ Conversion completed with issues:")
        print(f"  - Successful: {converted_count}")
        print(f"  - Failed: {failed_count}")
        return False

def main():
    """Main conversion function with command line support."""
    # Default paths based on standard CLASS-SZ data structure  
    default_input = os.path.expanduser("~/class_sz_data_directory/ede")
    default_output = os.path.expanduser("~/class_sz_data_directory/ede_v2_numpy")
    
    # Parse command line arguments
    if len(sys.argv) == 1:
        input_dir = default_input
        output_dir = default_output
    elif len(sys.argv) == 3:
        input_dir = sys.argv[1] 
        output_dir = sys.argv[2]
    else:
        print("Usage:")
        print("  python convert_ede_v2.py                    # Use default paths")
        print("  python convert_ede_v2.py <input> <output>   # Specify paths")
        print()
        print("Default paths:")
        print(f"  Input:  {default_input}")
        print(f"  Output: {default_output}")
        return
    
    # Verify TensorFlow is available for loading original files
    try:
        import tensorflow as tf
        print(f"Using TensorFlow {tf.__version__} for reading original files")
    except ImportError:
        print("Error: TensorFlow is required to read the original emulator files")
        print("Install with: pip install tensorflow")
        return
    
    # Run conversion
    success = convert_ede_v2_emulators(input_dir, output_dir)
    
    if success:
        print("\n🎉 Ready to use with hmfast:")
        print("from hmfast import EDEEmulator")
        print("emulator = EDEEmulator()  # Will use the converted files")
    else:
        print("\n❌ Conversion failed. Check error messages above.")

if __name__ == "__main__":
    main()