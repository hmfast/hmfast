"""
Utility functions for cosmological calculations.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from typing import Dict, Any
from functools import partial
jax.config.update("jax_enable_x64", True)


def interpolate_tracer(z, m, tracer, ell_eval):
    """
    Interpolate u_ell values onto a uniform ell grid for multiple m values. 
    """

    ells, u_ells = tracer.compute_u_ell(z, m) 

    # Interpolator function for a single m
    def interpolate_single(ell, u_ell):
        interpolator = jscipy.interpolate.RegularGridInterpolator((ell,), u_ell, method='linear', bounds_error=False, fill_value=None)
        return interpolator(ell_eval)

    # Vectorize the interpolation across all m and interpolate
    u_ell_eval = jax.vmap(interpolate_single, in_axes=(0, 0), out_axes=0)(ells, u_ells)

    return ell_eval, u_ell_eval

    
