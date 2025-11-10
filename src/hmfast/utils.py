"""
Physical constants and utilities for hmfast tSZ calculations.

Based on classy_szfast/utils.py to ensure consistent normalization.
"""

import numpy as np
import jax.numpy as jnp

# Physical constants - matching classy_szfast exactly
# ===================================================

# Basic constants
kb = 1.38064852e-23  # Boltzmann constant in m^2 kg s^-2 K^-1
clight = 299792458.  # speed of light in m/s
hplanck = 6.62607004e-34  # Planck constant in m^2 kg / s

# CMB temperature
firas_T0 = 2.728  # pivot temperature used in the Max Lkl Analysis
firas_T0_bf = 2.725  # best-fitting temperature
Tcmb_uk = 2.7255e6  # CMB temperature in μK

# Gravitational constant
G_newton = 6.674e-11  # Newton's constant in m^3 kg^-1 s^-2

# Critical density
rho_crit_over_h2_in_GeV_per_cm3 = 1.0537e-5

# Electron properties
me_in_eV = 510998.95  # electron mass in eV/c^2
sigmat_cm = 6.6524587321e-25  # Thomson cross section in cm^2
sigmat_over_mec2 = sigmat_cm / me_in_eV

# Unit conversions
Mpc_to_cm = 3.085677581e24  # 1 Mpc in cm

class Const:
    """
    Physical constants class matching classy_szfast structure.
    """
    c_km_s = 299792.458  # speed of light in km/s
    h_J_s = 6.626070040e-34  # Planck's constant
    kB_J_K = 1.38064852e-23  # Boltzmann constant

    _c_ = 2.99792458e8      # c in m/s 
    _Mpc_over_m_ = 3.085677581282e22  # conversion factor from meters to megaparsecs 
    _Gyr_over_Mpc_ = 3.06601394e2  # conversion factor from megaparsecs to gigayears
    _G_ = 6.67428e-11             # Newton constant in m^3/Kg/s^2 
    _eV_ = 1.602176487e-19        # 1 eV expressed in J 

    # parameters entering in Stefan-Boltzmann constant sigma_B 
    _k_B_ = 1.3806504e-23
    _h_P_ = 6.62606896e-34
    _M_sun_ = 1.98855e30  # solar mass in kg

# tSZ specific constants
# ======================
def mpc_per_h_to_cm(mpc_per_h, h):
    """
    Converts a distance in Mpc/h to centimeters.
    
    Parameters
    ----------
    mpc_per_h : float or array
        Distance in Mpc/h
    h : float
        Dimensionless Hubble parameter
        
    Returns
    -------
    float or array
        Distance in cm
    """
    # Convert Mpc/h to Mpc by dividing by h
    mpc = mpc_per_h / h
    # Convert Mpc to cm
    cm = mpc * Mpc_to_cm
    return cm

def get_ell_range(lmin=10.0, lmax=10000.0, dlogell=0.1):
    """
    Generate logarithmically spaced ell range for tSZ calculations.
    
    Parameters
    ----------
    lmin : float
        Minimum multipole
    lmax : float  
        Maximum multipole
    dlogell : float
        Logarithmic spacing
        
    Returns
    -------
    jnp.ndarray
        Array of ell values
    """
    log_ell_min = jnp.log10(lmin)
    log_ell_max = jnp.log10(lmax)
    n_ell = int((log_ell_max - log_ell_min) / dlogell) + 1
    log_ell_array = jnp.linspace(log_ell_min, log_ell_max, n_ell)
    ell_array = 10.0**log_ell_array
    return ell_array

def get_ell_binwidth(ell_array=None):
    """
    Get bin widths for ell array - simplified for now.
    
    Parameters
    ----------
    ell_array : jnp.ndarray, optional
        Array of ell values
        
    Returns
    -------
    jnp.ndarray
        Array of bin widths (currently returns 1.0 for all bins)
    """
    if ell_array is None:
        ell_array = get_ell_range()
    
    # Simplified - return array of ones for now
    # In full implementation would compute proper bin widths
    return jnp.ones_like(ell_array)

def simpson(y, x=None, dx=1.0, axis=0):
    """
    Simpson's rule integration - simplified JAX implementation.
    
    For now, falls back to trapezoidal rule which is more stable in JAX.
    
    Parameters
    ----------
    y : jnp.ndarray
        Values to integrate
    x : jnp.ndarray, optional
        Integration variable
    dx : float
        Grid spacing if x not provided
    axis : int
        Integration axis
        
    Returns
    -------
    jnp.ndarray
        Integrated result
    """
    # Use trapezoidal rule for stability in JAX
    if x is not None:
        return jnp.trapezoid(y, x, axis=axis)
    else:
        return jnp.trapezoid(y, dx=dx, axis=axis)

def compute_rho_crit_z0(h):
    """
    Compute critical density at z=0 from fundamental constants.
    
    Parameters
    ----------
    h : float
        Dimensionless Hubble parameter (H0/100)
        
    Returns
    -------
    float
        Critical density in M_sun h^2 / (Mpc/h)^3
    """
    # H0 in SI units: H0 = 100 * h km/s/Mpc
    H0_SI = 100.0 * h * 1e3 / Const._Mpc_over_m_  # Convert to 1/s
    
    # Critical density: rho_crit = 3 H0^2 / (8 π G)
    rho_crit_SI = 3.0 * H0_SI**2 / (8.0 * jnp.pi * Const._G_)  # kg/m^3
    
    # Convert to M_sun h^2 / (Mpc/h)^3
    # Factor of h^2 comes from: (Mpc/h)^3 -> Mpc^3 requires /h^3, M_sun/h -> M_sun requires /h
    # Net: (kg/m^3) * (m^3/Mpc^3) * (M_sun/kg) * h^3 / h = M_sun h^2 / (Mpc/h)^3
    rho_crit_cosmo = rho_crit_SI * (Const._Mpc_over_m_**3 / Const._M_sun_) * h**2
    
    return rho_crit_cosmo

def compute_r_delta(m_delta, delta, z, omega_m, h):
    """
    Compute r_delta from mass, overdensity, redshift and cosmology.
    
    Parameters
    ----------
    m_delta : float or array
        Halo mass in M_sun/h
    delta : float
        Overdensity parameter (e.g., 200, 500)
    z : float  
        Redshift
    omega_m : float
        Matter density parameter
    h : float
        Dimensionless Hubble parameter
        
    Returns
    -------
    float or array
        r_delta in Mpc/h
    """
    # Critical density at z=0
    rho_crit_0 = compute_rho_crit_z0(h)
    
    # Mean density at redshift z
    rho_mean_z = rho_crit_0 * omega_m * (1.0 + z)**3
    
    # r_delta = (3 M_delta / (4π δ ρ_mean(z)))^(1/3)
    r_delta = ((3.0 * m_delta) / (4.0 * jnp.pi * delta * rho_mean_z))**(1.0/3.0)
    
    return r_delta


# Legacy cosmology utils (keeping for backward compatibility)
# ===========================================================

from typing import Dict, Any
from functools import partial
import jax

class cosmology_utils:
    """Utility functions for cosmological calculations."""
    
    @staticmethod
    @jax.jit
    def hubble_parameter(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute Hubble parameter H(z).
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            H(z) in units of H0
        """
        Omega_m = cosmology.get('Omega_m', 0.3153)
        Omega_L = 1.0 - Omega_m  # Flat universe assumption
        
        return jnp.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    
    @staticmethod
    @jax.jit
    def comoving_distance(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute comoving distance to redshift z.
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            Comoving distance in Mpc/h
        """
        # Simple approximation - in practice would integrate
        h = cosmology.get('h', 0.6736)
        c_km_s = 299792.458  # km/s
        H0 = 100 * h  # km/s/Mpc
        
        # Approximate integral for flat LCDM
        Omega_m = cosmology.get('Omega_m', 0.3153)
        Ez_inv_approx = 1.0 / jnp.sqrt(Omega_m * (1 + z)**3 + (1 - Omega_m))
        
        return c_km_s / H0 * z * Ez_inv_approx
    
    @staticmethod
    @jax.jit
    def growth_factor(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute linear growth factor D(z).
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            Growth factor normalized to 1 at z=0
        """
        # Approximate growth factor for flat LCDM
        Omega_m = cosmology.get('Omega_m', 0.3153)
        a = 1.0 / (1.0 + z)
        
        # Carroll, Press & Turner 1992 approximation
        omega_a = Omega_m / (Omega_m + (1 - Omega_m) * a**3)
        
        growth = 2.5 * omega_a / (omega_a**(4./7.) - (1 - omega_a) + 
                                  (1 + omega_a/2.) * (1 + (1 - omega_a)/70.))
        
        # Normalize to z=0
        omega_0 = Omega_m
        growth_0 = 2.5 * omega_0 / (omega_0**(4./7.) - (1 - omega_0) + 
                                     (1 + omega_0/2.) * (1 + (1 - omega_0)/70.))
        
        return growth / growth_0 * a
    
    @staticmethod
    @jax.jit
    def sigma8_z(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute sigma_8 at redshift z.
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            sigma_8(z)
        """
        sigma8_0 = cosmology.get('sigma_8', 0.8111)
        D_z = cosmology_utils.growth_factor(z, cosmology)
        
        return sigma8_0 * D_z
    
    @staticmethod
    @jax.jit
    def critical_density(z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute critical density at redshift z.
        
        Parameters
        ----------
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            Critical density in Msun h^2 / Mpc^3
        """
        h = cosmology.get('h', 0.6736)
        H_z = cosmology_utils.hubble_parameter(z, cosmology) * 100 * h  # km/s/Mpc
        
        # Critical density in kg/m^3
        G = 6.67430e-11  # m^3/kg/s^2
        H_z_SI = H_z * 1e3 / (3.086e22)  # 1/s
        rho_crit = 3 * H_z_SI**2 / (8 * jnp.pi * G)  # kg/m^3
        
        # Convert to Msun h^2 / Mpc^3
        Msun = 1.989e30  # kg
        Mpc = 3.086e22  # m
        
        return rho_crit * (Mpc**3 / Msun) * h**2
    
    @staticmethod
    @jax.jit  
    def virial_radius(M: float, z: float, cosmology: Dict[str, float]) -> float:
        """
        Compute virial radius for a halo of mass M.
        
        Parameters
        ----------
        M : float
            Halo mass in Msun/h
        z : float
            Redshift
        cosmology : dict
            Cosmological parameters
            
        Returns
        -------
        float
            Virial radius in Mpc/h
        """
        Delta_vir = 200.0  # Typical overdensity definition
        rho_crit = cosmology_utils.critical_density(z, cosmology)
        Omega_m = cosmology.get('Omega_m', 0.3153)
        rho_m = Omega_m * rho_crit * (1 + z)**3
        
        # R_vir = (3M / (4π * Delta_vir * rho_m))^(1/3)
        return (3 * M / (4 * jnp.pi * Delta_vir * rho_m))**(1./3.)