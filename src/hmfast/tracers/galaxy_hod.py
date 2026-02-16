import os
import numpy as np # it may be a good idea to eventually remove numpy dependence altogether, but now we need it for np.loadtxt
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax.scipy.special import sici, erf 
from hmfast.tracers.base_tracer import BaseTracer
from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.defaults import merge_with_defaults
from hmfast.download import get_default_data_path
from hmfast.literature import c_D08

jax.config.update("jax_enable_x64", True)



class GalaxyHODTracer(BaseTracer):
    """
    Galaxy HOD tracer implementing central + satellite occupation and
    NFW satellites. Implements the formalism described in Kusiak et al (2023)
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
        self.halo_model = HaloModel(cosmo_model=cosmo_model)  # Eventually want to allow the user to pass hmf prescription (e.g. T08)

    def load_dndz_data(self):
        dndz_path = os.path.join(get_default_data_path(), "auxiliary_files", "normalised_dndz_cosmos_0.txt")
        data = np.loadtxt(dndz_path)
        return jnp.array(data)
        


    def get_N_centrals(self, m, params = None):
        """Mean central occupation: shape = M.shape"""

        params = merge_with_defaults(params)
        M_min = params["M_min_HOD"]
        sigma = params["sigma_log10M_HOD"]

        # Set up the input x of the error function and evaluate it
        x = (jnp.log10(m) - jnp.log10(M_min)) / sigma
        return 0.5 * (1.0 + erf(x))

    
    def get_N_satellites(self, m, params = None):
        """Mean satellite occupation: shape = M.shape"""
        params = merge_with_defaults(params)
        M0 = params["M0_HOD"]
        M1p = params["M1_prime_HOD"]
        alpha = params["alpha_s_HOD"]

        # power law only above M0 and use jnp.where to keep differentiability
        pow_term = jnp.maximum((m - M0) / M1p, 0.0)**alpha
        N_c = self.get_N_centrals(m, params = params)
        return  N_c * pow_term
        

    def get_ng_bar(self, z, m, params = None):
        """
        Compute comoving galaxy number density ng(z) = ∫ dlnM [dn/dlnM] [Nc+Ns].
        halo_model: HaloModel instance
        tracer: GalaxyHODTracer instance (provides HOD via helper funcs)
        z: scalar redshift
        params: parameter dict 
        """
        
        params = merge_with_defaults(params)
        logm = jnp.log(m)
        z = jnp.atleast_1d(z)

        Nc = self.get_N_centrals(m, params=params)
        Ns = self.get_N_satellites(m, params=params)
        Ntot = Nc + Ns
    
        def ng_bar_single(z_single):
            dndlnm = self.halo_model.halo_mass_function(z_single, m, params=params)  # shape (n_m,)
            integrand = dndlnm * Ntot
            return jnp.trapezoid(integrand, x=logm)
    
        # vectorize over z
        return jax.vmap(ng_bar_single)(z)
        

    def get_W_g(self, z, params=None):
        """
        Return Wg_grid at requested z.
        Uses pre-loaded dndz_data = [z, phi_prime].
        """
        
        params = merge_with_defaults(params)
        zq = jnp.atleast_1d(jnp.array(z, dtype=jnp.float64))
    
        # Extract precomputed phi_prime
        dndz_data = self.load_dndz_data()
        z_data = dndz_data[:, 0]
        phi_prime_data = dndz_data[:, 1]
    
        # Interpolate phi_prime to requested z
        phi_prime_at_z = jnp.interp(zq, z_data, phi_prime_data, left=0.0, right=0.0)
    
        H_grid = self.emulator.hubble_parameter(zq, params=params)  # 1/Mpc
        chi_grid = self.emulator.angular_diameter_distance(zq, params=params) * (1.0 + zq)  # Mpc comov

        # Assemble W_g on the grid
        W_g = H_grid * (phi_prime_at_z / chi_grid**2)
        return W_g



    def get_u_ell(self, z, m, moment=1, params=None):
        """ 
        Compute either the first or second moment of the galaxy HOD tracer u_ell.
        For galaxy HOD:, 
            First moment:     W_g / ng_bar * [Nc + Ns * u_ell_m]
            Second moment:    W_g^2 / ng_bar^2 * [Ns^2 * u_ell_m^2 + 2 * Ns * u_ell_m]
        You cannot simply take u_ell_g**2.
        """

        params = merge_with_defaults(params)
        Ns = self.get_N_satellites(m, params=params)
        Nc = self.get_N_centrals(m, params=params)
        ng = self.get_ng_bar(z, m, params=params) * (params["H0"]/100)**3
        W  = self.get_W_g(z, params=params)

        # Compute u_m_ell from BaseTracer
        ell, u_m = self.u_ell_analytic(z, m, params=params)  
    
        moment_funcs = [
            lambda _: (W/ng)[:, None] * (Nc[:, None] + Ns[:, None] * u_m),
            lambda _: (W**2/ng**2)[:, None] * (Ns[:, None]**2 * u_m**2 + 2*Ns[:, None] * u_m),
        ]
    
        u_ell = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return ell, u_ell
  
        