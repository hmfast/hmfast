"""
HMFast: Machine learning accelerated and differentiable halo model code.

This package provides fast, differentiable halo model calculations using JAX
and machine learning emulators for cosmological applications.
"""

__version__ = "0.1.0"
__author__ = "Boris"
__email__ = "boris@example.com"

from .ede_emulator import EDEEmulator
from .clean_nn_emulator import CleanRestoreNN, CleanRestorePCAplusNN
from .utils import cosmology_utils

__all__ = ["EDEEmulator", "CleanRestoreNN", "CleanRestorePCAplusNN", "cosmology_utils"]