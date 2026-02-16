import os
import numpy as np # it may be a good idea to eventually remove numpy dependence altogether, but now we need it for np.loadtxt
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.scipy.special import sici
from hmfast.tracers.base_tracer import BaseTracer
from hmfast.emulator import Emulator
from hmfast.defaults import merge_with_defaults
from hmfast.download import get_default_data_path
from hmfast.literature import c_D08

jax.config.update("jax_enable_x64", True)



class GalaxyLensingTracer(BaseTracer):
    """
    Galaxy lensing tracer. Implements the formalism described in Kusiak et al (2023)
    Link to paper: https://arxiv.org/pdf/2203.12583

    Parameters
    ----------
    emulator : 
        Cosmological emulator used to compute cosmological quantities
     x : array
        The x array used to define the radial profile over which the tracer will be evaluated
    """

    def __init__(self, cosmo_model=0, concentration_relation=c_D08):        
        
        self.concentration_relation = concentration_relation

        # Load emulator and make sure the required files are loaded outside of jitted functions
        self.emulator = Emulator(cosmo_model=cosmo_model)
        self.emulator._load_emulator("DAZ")
        self.emulator._load_emulator("HZ")

    
    def load_file_data(self, filename):
        """
        Load a two-column redshift distribution file and return the data as a jax array
    
        Assumes the file contains two whitespace-separated columns: z  f(z).
        If `filename` is not an absolute path, it is resolved relative to HMFAST_DATA_PATH.
        """
        if not os.path.isabs(filename):
            filename = os.path.join(get_default_data_path(), "auxiliary_files", filename)

        data = np.loadtxt(filename)
        return jnp.array(data)

    
    def get_I_g(self, z, params=None):
        """
        Return Wg_grid at requested z.
        Uses pre-loaded dndz_data = [z, phi_prime].
        Integrates only over sources behind the lens (z_s > z).
        """
        
        params = merge_with_defaults(params)
        zq = jnp.atleast_1d(jnp.array(z, dtype=jnp.float64))
        h = params["H0"] / 100
        
        # Load source distribution
        dndz_data = self.load_file_data(filename="nz_source_normalized_bin4.txt")
        z_data = dndz_data[:, 0]
        phi_prime_data = dndz_data[:, 1]
    
        # Load lens redshift distribution
        nz_lens_data = self.load_file_data(filename="nz_lens_bin1.txt")
        z_s_data = nz_lens_data[:, 0]
        n_s_data = nz_lens_data[:, 1]
    
        # Interpolate phi_prime to z_s grid
        phi_prime_at_z_s = jnp.interp(z_s_data, z_data, phi_prime_data, left=0.0, right=0.0)
    
        # Angular distances
        chi_z_s = self.emulator.angular_diameter_distance(z_s_data) * (1 + z_s_data) * h
        chi_z = self.emulator.angular_diameter_distance(zq) * (1 + zq) * h
    
        # Reshape for broadcasting
        chi_z_s = chi_z_s[:, None]  # (N_s, 1)
        chi_z = chi_z[None, :]      # (1, N_z)
    
        # Lensing factor
        chi_diff = (chi_z_s - chi_z) / chi_z_s
    
        # Mask: only include sources behind the lens
        mask = (z_s_data[:, None] > zq[None, :])  # (N_s, N_z)
        chi_diff_masked = chi_diff * mask
    
        # Integrate over z_s using trapezoid
        I_g = jnp.trapezoid(phi_prime_at_z_s[:, None] * chi_diff_masked, x=z_s_data, axis=0)
    
        return I_g



    def get_W_kappa_g(self, z, params=None):
        """
        Compute the galaxy lensing kernel W_kappa_g at redshift z.
        """
        # Merge default parameters with input
        params = merge_with_defaults(params)
        cparams = self.emulator.get_all_cosmo_params(params=params)
        zq = jnp.atleast_1d(jnp.array(z, dtype=jnp.float64))  # Ensure z is an array

        c_km_s = 299792.458  # Speed of light in km/s
       
        # Cosmological constants
        H0 = params["H0"]  # Hubble constant in km/s/Mpc
        h = cparams["h"]
        Omega_m = cparams["Omega0_m"]  # Matter density parameter

        # Compute comoving distance and Hubble parameter
        chi_z = self.emulator.angular_diameter_distance(zq, params=params) * (1 + zq) * h # Comoving distance in Mpc/h
        H_z = self.emulator.hubble_parameter(zq, params=params)   # Hubble parameter in km/s/Mpc
    
        I_g = self.get_I_g(zq, params=params) 
    
        # Compute the CMB lensing kernel
        W_kappa_g =  (
            (3.0 / 2.0) * Omega_m * 
            (H0/c_km_s)**2 / h**2 *
            (1 + z) / chi_z  *
            I_g 
        ) 
    
        return W_kappa_g 

    

    def get_u_ell(self, z, m, moment=1, params=None):
        """ 
        Compute either the first or second moment of the CMB lensing tracer u_ell.
        For galaxy lensing:, 
            First moment:     W_k_g * u_ell_m
            Second moment:    W_k_g^2 * u_ell_m^2 
        """

        params = merge_with_defaults(params)
        cparams = self.emulator.get_all_cosmo_params(params)
        W = self.get_W_kappa_g(z, params=params) 

        # Compute u_m_ell from BaseTracer
        ell, u_m = self.u_ell_analytic(z, m, params=params)

        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"] / cparams["h"]**2  
        m_over_rho_mean = (m / rho_mean_0)[:, None]  # shape (N_m, 1)
        m_over_rho_mean = jnp.broadcast_to(m_over_rho_mean, u_m.shape)
        u_m *= m_over_rho_mean
    
        moment_funcs = [
            lambda _:  W[:, None] * u_m,
            lambda _: (W**2)[:, None] * u_m**2,
        ]
    
        u_ell = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return ell, u_ell
  
  


       
