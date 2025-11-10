#!/usr/bin/env python3
"""
tSZ Power Spectrum Calculation for hmfast

This module implements the 1-halo term tSZ power spectrum C_l^yy calculation
following the patterns from tszsbi/tszpower/tsz.py, using:
- GNFW pressure profiles from profiles.py
- HMF from hmfast implementation  
- Limber integral with comoving volume
"""

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import functools
import mcfit
from mcfit import Hankel
from .ede_emulator import EDEEmulator
from .utils import (
    me_in_eV, sigmat_cm, sigmat_over_mec2, Mpc_to_cm, 
    get_ell_range, get_ell_binwidth, simpson, mpc_per_h_to_cm,
    compute_rho_crit_z0, compute_r_delta, Const
)

# JAX configuration
jax.config.update("jax_enable_x64", True)

@jax.jit
def _compute_clyy_core_compiled(integrand, z_grid, m_grid):
    """
    Compiled core integration function following tsz.py strategy.
    Uses lax.scan for efficiency like the original.
    """
    from jax import lax
    
    logm_grid = jnp.log(m_grid)
    n_ell = integrand.shape[2]
    
    # Define scan body function following tsz.py exactly
    def scan_body(_, i):
        # For the i-th ell value, extract the corresponding slice
        integrand_i = integrand[:, :, i]  # Shape: (n_z, n_m)
        
        # First integrate over mass (axis=1)
        partial_m = jnp.trapezoid(integrand_i, logm_grid, axis=1)  # Shape: (n_z,)
        
        # Then integrate over redshift (axis=0)
        result = jnp.trapezoid(partial_m, z_grid, axis=0)  # Scalar
        
        return None, result
    
    # Use lax.scan over ell indices like tsz.py
    _, C_yy = lax.scan(scan_body, None, jnp.arange(n_ell))
    
    return C_yy

class TSZPowerSpectrum:
    """
    Class for computing tSZ power spectrum C_l^yy using the 1-halo term.
    
    This follows the tsz.py implementation but adapted for hmfast.
    """
    
    def __init__(self):
        """Initialize TSZ power spectrum calculator."""
        self.emulator = EDEEmulator()
        
        # Precompute the x grid for Hankel transform (match tszsbi pattern)
        self._X_GRID = jnp.logspace(jnp.log10(1e-6), jnp.log10(1e6), num=2048*2)
        
        # Construct the Hankel transform using the precomputed x grid
        self._Hankel = Hankel(self._X_GRID, nu=0.5, lowring=True, backend='jax')
        self._Hankel_jit = jax.jit(functools.partial(self._Hankel, extrap=False))
        
    def get_ell_grid(self, lmin=10.0, lmax=1000.0, dlogell=0.1):
        """
        Generate ell grid with specified parameters.
        
        Parameters
        ----------
        lmin : float
            Minimum multipole
        lmax : float  
            Maximum multipole
        dlogell : float
            Logarithmic spacing in ell
            
        Returns
        -------
        jnp.ndarray
            Array of multipole values
        """
        log10_lmin = jnp.log10(lmin)
        log10_lmax = jnp.log10(lmax)
        num_points = int((log10_lmax - log10_lmin) / dlogell) + 1
        return jnp.logspace(log10_lmin, log10_lmax, num=num_points)
    
    def dVdzdOmega(self, z, params_values_dict=None):
        """
        Comoving volume element dV/dz/dOmega following tsz.py pattern.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict, optional
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Volume element in (Mpc/h)^3 sr^-1
        """
        rparams = self.emulator.get_all_relevant_params(params_values_dict)
        h = rparams['h']
        
        # Angular diameter distance in Mpc/h
        dAz = self.emulator.get_angular_distance_at_z(z, params_values_dict=params_values_dict) * h
        
        # Hubble parameter in (Mpc/h)^-1
        Hz = self.emulator.get_hubble_at_z(z, params_values_dict=params_values_dict) / h
        
        return (1 + z)**2 * dAz**2 / Hz
    
    def gnfw_pressure_profile(self, x, z, m, params_values_dict=None):
        """
        GNFW pressure profile following profiles.py pattern.
        
        Parameters
        ----------
        x : jnp.ndarray
            Scaled radius r/r_delta
        z : float
            Redshift
        m : float or jnp.ndarray  
            Mass in M_sun/h
        params_values_dict : dict, optional
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Pressure profile in eV cm^-3
        """
        rparams = self.emulator.get_all_relevant_params(params_values_dict)
        
        # Physical constants and conversions
        conv_fac = 299792.458  # speed of light km/s
        h = rparams['h']
        
        # Hubble parameters
        H = self.emulator.get_hubble_at_z(z, params_values_dict=params_values_dict) * conv_fac
        H0 = rparams['H0']
        
        # GNFW parameters (using default values following tszsbi pattern)
        B = rparams.get('B', 1.0)  # Mass calibration factor
        m_delta_tilde = m / B
        
        # GNFW prefactor following exact tszsbi pattern
        C = 1.65 * (h / 0.7)**2 * (H / H0)**(8/3) * (m_delta_tilde / (0.7 * 3e14))**(2/3 + 0.12) * (0.7/h)**1.5
        
        # GNFW profile parameters  
        c500 = rparams.get('c500', 1.81)
        gamma = rparams.get('gammaGNFW', 0.31) 
        alpha = rparams.get('alphaGNFW', 1.33)
        beta = rparams.get('betaGNFW', 4.13)
        P0 = rparams.get('P0GNFW', 6.41)
        
        # Calculate scaled radius and pressure profile
        scaled_x = c500 * x
        term1 = scaled_x**(-gamma)
        term2 = (1 + scaled_x**alpha)**((gamma - beta) / alpha)
        Pe = C * P0 * term1 * term2
        
        return Pe
    
    def window_function(self, x, x_min=1e-6, x_max=20.0):
        """Window function for integration limits."""
        return jnp.where((x >= x_min) & (x <= x_max), 1.0, 0.0)
    
    def hankel_integrand(self, x, z, m, x_min=1e-6, x_max=20.0, params_values_dict=None):
        """
        Compute x^0.5 * Pe(x) * W(x) for Hankel transform.
        
        Parameters
        ----------
        x : jnp.ndarray
            Scaled radius array
        z : float
            Redshift  
        m : jnp.ndarray
            Mass array
        x_min, x_max : float
            Integration limits
        params_values_dict : dict, optional
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Integrand array with shape (len(m), len(x))
        """
        def single_m(m_val):
            Pe = self.gnfw_pressure_profile(x, z, m_val, params_values_dict=params_values_dict)
            W_x = self.window_function(x, x_min, x_max)
            return x**0.5 * Pe * W_x
        
        # Vectorize over mass array
        result = jax.vmap(single_m)(m)
        return result
    
    def mpc_per_h_to_cm(self, mpc_per_h, h):
        """Convert Mpc/h to cm."""
        Mpc_to_cm = 3.085677581e24  # cm per Mpc
        return (mpc_per_h / h) * Mpc_to_cm
    
    def y_ell_prefactor(self, z, m, delta=500, params_values_dict=None):
        """
        Prefactor for y_ell calculation.
        
        Parameters
        ----------
        z : float
            Redshift
        m : jnp.ndarray
            Mass array in M_sun/h
        delta : float
            Overdensity parameter
        params_values_dict : dict, optional
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Prefactor array
        """
        rparams = self.emulator.get_all_relevant_params(params_values_dict)
        h = rparams['h']
        B = rparams.get('B', 1.0)
        
        # Physical constants
        me_in_eV = 510998.95  # electron mass in eV/c^2
        sigmat_cm = 6.6524587321e-25  # Thomson cross section in cm^2
        sigmat_over_mec2 = sigmat_cm / me_in_eV
        
        # Distances and scales
        dAz = self.emulator.get_angular_distance_at_z(z, params_values_dict=params_values_dict) * h
        
        # Get r_delta using critical density at z=0
        # r_delta = (3M / (4π * δ * ρ_crit * Ω_m))^(1/3)
        # Using critical density at z=0: ρ_crit = 2.78e11 h^2 M_sun/h per (Mpc/h)^3
        rho_crit = compute_rho_crit_z0(h)  # Use proper calculation from utils
        Omega_m = rparams['Omega0_m']
        rho_mean = rho_crit * Omega_m
        
        # r_delta for overdensity δ with respect to mean density
        r_delta = ((3.0 * m) / (4.0 * jnp.pi * delta * rho_mean))**(1.0/3.0) / (B**(1.0/3.0))
        
        ell_delta = dAz / r_delta
        r_delta_cm = self.mpc_per_h_to_cm(r_delta, h)
        
        prefactor = sigmat_over_mec2 * 4 * jnp.pi * r_delta_cm / (ell_delta**2)
        
        return prefactor
    
    def y_ell_complete(self, z, m, x_min=1e-6, x_max=20.0, params_values_dict=None):
        """
        Complete y_ell calculation using Hankel transform.
        
        Parameters
        ----------
        z : float
            Redshift
        m : jnp.ndarray
            Mass array
        x_min, x_max : float
            Integration limits
        params_values_dict : dict, optional
            Cosmological parameters
            
        Returns
        -------
        tuple
            (ell_array, y_ell_array) with shapes (n_mass, n_ell) each
        """
        rparams = self.emulator.get_all_relevant_params(params_values_dict)
        h = rparams['h']
        B = rparams.get('B', 1.0)
        
        # Get prefactor
        prefactor = self.y_ell_prefactor(z, m, params_values_dict=params_values_dict)
        
        # Compute Hankel integrand
        integrand = self.hankel_integrand(
            self._X_GRID, z, m, x_min=x_min, x_max=x_max, 
            params_values_dict=params_values_dict
        )
        
        # Apply Hankel transform
        k, y_k = self._Hankel_jit(integrand)  # k = ell/ell_delta
        
        # Calculate angular scales
        dAz = self.emulator.get_angular_distance_at_z(z, params_values_dict=params_values_dict) * h
        
        # Approximate r_delta calculation - use same as prefactor
        delta = 500
        rho_crit = compute_rho_crit_z0(h)  # Use proper calculation from utils
        Omega_m = rparams['Omega0_m']
        rho_mean = rho_crit * Omega_m
        r_delta = ((3.0 * m) / (4.0 * jnp.pi * delta * rho_mean))**(1.0/3.0) / (B**(1.0/3.0))
        
        ell_delta = dAz / r_delta
        
        # Compute final ell and y_ell arrays
        ell = k[None, :] * ell_delta[:, None]  # Shape: (n_mass, n_ell)
        y_ell = prefactor[:, None] * y_k * jnp.sqrt(jnp.pi / (2 * k[None, :]))
        
        return ell, y_ell
    
    def y_ell_interpolate(self, z, m, ell_grid, params_values_dict=None):
        """
        Interpolate y_ell onto specified ell grid.
        
        Parameters
        ----------
        z : float
            Redshift
        m : jnp.ndarray
            Mass array
        ell_grid : jnp.ndarray
            Target ell values
        params_values_dict : dict, optional
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Interpolated y_ell values with shape (n_mass, n_ell)
        """
        # Get complete y_ell calculation
        ell_native, y_ell_native = self.y_ell_complete(z, m, params_values_dict=params_values_dict)
        
        # Interpolate each mass onto the target grid
        def interpolate_single(ell_in, y_ell_in):
            interpolator = jscipy.interpolate.RegularGridInterpolator(
                (ell_in,), y_ell_in, method='linear', bounds_error=False, fill_value=0.0
            )
            return interpolator(ell_grid)
        
        # Vectorize over masses
        interpolate_all = jax.vmap(interpolate_single, in_axes=(0, 0), out_axes=0)
        y_ell_interp = interpolate_all(ell_native, y_ell_native)
        
        return y_ell_interp
    
    @jax.jit
    def simpson_integration(self, y, x, axis=0):
        """Fast Simpson's rule integration - JAX compiled."""
        # Use trapezoidal rule for speed (Simpson's rule overhead not worth it for this precision)
        return jnp.trapezoid(y, x, axis=axis)
    
    def _get_integral_grid(self, ell_grid, z_grid, m_grid, params_values_dict):
        """
        Pre-compute integrand grid following tsz.py strategy.
        This is the expensive part - done once per parameter set.
        """
        # Get y_ell profiles for each redshift (expensive Hankel transforms)
        def get_yell_for_z(zp):
            return self.y_ell_interpolate(zp, m_grid, ell_grid, params_values_dict=params_values_dict)
        
        y_ell_mz = jax.vmap(get_yell_for_z)(z_grid)
        # Shape: (n_z, n_m, n_ell)
        
        # Get HMF for all redshifts (fast)
        def get_hmf_for_z(zp):
            return self.emulator.get_hmf_at_z_and_m(zp, m_grid, delta=500, delta_def='critical', params_values_dict=params_values_dict)
        
        dndlnm_mz = jax.vmap(get_hmf_for_z)(z_grid)
        # Shape: (n_z, n_m)
        
        # Get comoving volume (fast)
        comov_vol = jax.vmap(lambda zp: self.dVdzdOmega(zp, params_values_dict))(z_grid)
        # Shape: (n_z,)
        
        # Construct integrand: y_ell^2 * dndlnm * dV/dz/dOmega
        y_ell_sq = y_ell_mz**2
        dndlnm_expanded = dndlnm_mz[:, :, None]  # Shape: (n_z, n_m, 1)
        comov_vol_expanded = comov_vol[:, None, None]  # Shape: (n_z, 1, 1)
        
        # Final integrand shape: (n_z, n_m, n_ell)
        integrand = y_ell_sq * dndlnm_expanded * comov_vol_expanded
        
        return integrand
    
    def _compute_clyy_core(self, integrand, z_grid, m_grid):
        """
        Core integration function following tsz.py strategy.
        Uses lax.scan for efficiency like the original.
        """
        # Use the static compiled function
        return _compute_clyy_core_compiled(integrand, z_grid, m_grid)

    def compute_clyy_fast(self, params_values_dict=None, lmin=10.0, lmax=10000.0, dlogell=0.1,
                          z_min=0.01, z_max=3.0, M_min=1e13, M_max=1e16, n_z=100, n_m=100):
        """
        FAST tSZ power spectrum computation - matches tsz.py strategy exactly.
        
        Uses 100×100 redshift/mass grid for proper cosmological accuracy.
        Two-phase computation: expensive grid pre-computation + fast integration.
        """
        import time
        
        # Set up grids (production quality defaults: 100×100)
        ell_grid = self.get_ell_grid(lmin, lmax, dlogell)
        z_grid = jnp.geomspace(z_min, z_max, n_z)
        m_grid = jnp.geomspace(M_min, M_max, n_m)
        
        print(f"Fast C_l^yy: {len(ell_grid)} ell × {n_z} z × {n_m} masses")
        
        # Time the computation
        start = time.time()
        
        # Phase 1: Pre-compute integrand grid (expensive - Hankel transforms)
        integrand = self._get_integral_grid(ell_grid, z_grid, m_grid, params_values_dict)
        
        # Phase 2: Fast integration (JAX compiled with lax.scan)
        C_yy = self._compute_clyy_core(integrand, z_grid, m_grid)
        
        elapsed = time.time() - start
        print(f"✓ Computed in {elapsed:.3f}s")
        
        return ell_grid, C_yy

    def compute_clyy(self, params_values_dict=None, lmin=10.0, lmax=10000.0, dlogell=0.1,
                     z_min=0.01, z_max=3.0, M_min=1e13, M_max=1e16, n_z=100, n_m=100):
        """
        Standard tSZ power spectrum computation - matches tsz.py strategy exactly.
        
        Uses 100×100 redshift/mass grid for cosmological accuracy.
        Two-phase computation: expensive grid pre-computation + fast integration.
        """
        # Set up grids
        ell_grid = self.get_ell_grid(lmin, lmax, dlogell)
        z_grid = jnp.geomspace(z_min, z_max, n_z)
        m_grid = jnp.geomspace(M_min, M_max, n_m)
        
        print(f"Computing C_l^yy on grid: {len(ell_grid)} ell × {n_z} z × {n_m} masses")
        
        # Phase 1: Pre-compute integrand grid (expensive - Hankel transforms)
        integrand = self._get_integral_grid(ell_grid, z_grid, m_grid, params_values_dict)
        
        # Phase 2: Fast integration (JAX compiled with lax.scan)
        C_yy = self._compute_clyy_core(integrand, z_grid, m_grid)
        
        return ell_grid, C_yy