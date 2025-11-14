import jax
import jax.numpy as jnp
import functools
import mcfit
from abc import ABC, abstractmethod


class HankelTransform:
    """
    Reusable Hankel transform wrapper for JAX-based computation.
    """
    def __init__(self, x_min=1e-6, x_max=1e6, x_npoints=4096, nu=0.5):
        self._x_grid = jnp.logspace(jnp.log10(x_min), jnp.log10(x_max), num=x_npoints)
        self._hankel = mcfit.Hankel(self._x_grid, nu=nu, lowring=True, backend='jax')
        self._hankel_jit = jax.jit(functools.partial(self._hankel, extrap=False))

    def transform(self, f_theta):
        """
        Perform the Hankel transform on a profile sampled on self._x_grid
        """
        k, y_k = self._hankel_jit(f_theta)
        return k, y_k


class BaseTracer(ABC):
    """
    Abstract base class for cosmological tracers.

    Sets up a radial grid and Hankel transform. Subclasses must implement
    `_get_hankel_inputs` to provide tracer-specific prefactors, integrands,
    and scale factors for computing u_ell.
    """
    
    def __init__(self, params):
        """
        Initialize the radial grid and Hankel transform.
        """
        self.params = params
        x_min, x_max, x_npoints = params['x_min'], params['x_max'], params['x_npoints']
        self.x_grid = jnp.logspace(jnp.log10(x_min), jnp.log10(x_max), x_npoints)
        self.hankel = HankelTransform(x_min=x_min, x_max=x_max, x_npoints=x_npoints, nu=0.5)

    @abstractmethod
    def _get_hankel_inputs(self, z, m):
        """Return (prefactor, hankel_integrand, scale_factor) arrays for given z, m."""
        pass

    def compute_u_ell(self, z, m):
        """
        Compute u_ell for this tracer at given z and m.
        """
        prefactor, integrand, scale_factor = self._get_hankel_inputs(z, m)
        k, u_k = self.hankel._hankel_jit(integrand)
        ell = k[None, :] * scale_factor[:, None]
        u_ell = prefactor[:, None] * u_k * jnp.sqrt(jnp.pi / (2 * k[None, :]))
        return ell, u_ell


