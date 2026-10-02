"""
HMFast: Machine learning accelerated and differentiable halo model code.

This package provides fast, differentiable halo model calculations using JAX
and machine learning emulators for cosmological applications.
"""

__version__ = "0.2.0"
__author__ = "Patrick Janulewicz, Licong Xu, Boris Bolliet"
__email__ = "pj407@cam.ac.uk"


from .download import download_emulators

download_emulators(emulator_set=["lcdm:v1", "ede:v2"], skip_existing=True)


import sys as _sys

from . import halos
from . import cosmology
from . import tracers

# Keep the pre-package import path hmfast.emulator_load working.
from .cosmology import emulator_load
_sys.modules[__name__ + ".emulator_load"] = emulator_load


__all__ = ["cosmology", "halos", "tracers", "download_emulators"]