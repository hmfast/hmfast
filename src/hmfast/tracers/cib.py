import jax
import jax.numpy as jnp

from hmfast.emulator import Emulator
from hmfast.halo_model import HaloModel
from hmfast.tracers.base_tracer import BaseTracer
from hmfast.defaults import merge_with_defaults
from hmfast.halo_model.mass_function import TW10SubHaloMass
from hmfast.utils import lambertw, Const

jax.config.update("jax_enable_x64", True)


class CIBTracer(BaseTracer):
    """
    CIB lensing tracer. 

    Parameters
    ----------
    emulator : 
        Cosmological emulator used to compute cosmological quantities
     x : array
        The x array used to define the radial profile over which the tracer will be evaluated
    """

    def __init__(self, halo_model, nu=100, subhalo_mass_function=TW10SubHaloMass()):        

        self.nu = nu
        self.subhalo_mass_function = subhalo_mass_function # Might eventually want to move this to halo_model

        # Load halo model with instantiated emulator and make sure the required files are loaded outside of jitted functions
        self.halo_model = halo_model
        self.halo_model.emulator._load_emulator("DAZ")
        self.halo_model.emulator._load_emulator("HZ")


    
    def sigma(self, z, m, nu, params=None):
        params = merge_with_defaults(params)

        M_eff_cib = params['m_eff_cib']
        sigma2_LM_cib = params['sigma2_LM_cib'] 

        # Log-normal in mass
        log10_m = jnp.log10(m)
        log10_M_eff = jnp.log10(M_eff_cib)
        Sigma_M = m / jnp.sqrt(2 * jnp.pi * sigma2_LM_cib)  *  jnp.exp( -(log10_m - log10_M_eff)**2 / (2 * sigma2_LM_cib) )
        return Sigma_M


    def phi(self, z, m, nu, params=None):
        params = merge_with_defaults(params)
        delta_cib = params["delta_cib"]
        z_p = params["z_plateau_cib"]

        Phi_z = jnp.where(z < z_p, (1 + z) ** delta_cib, 1.0)

        return Phi_z


    def theta(self, z, m, nu, params=None):

       
        """Spectral energy distribution function Theta(nu,z) for CIB, analogous to class_sz."""
        params = merge_with_defaults(params)
        T0 = params["T0_cib"]
        alpha_cib = params["alpha_cib"]
        beta_cib = params["beta_cib"]
        gamma_cib = params["gamma_cib"]
    
        h = Const._h_P_  # Planck [J s]
        k_B = Const._k_B_ #1.380649e-23  # Boltzmann [J/K]
        c = Const._c_  #2.99792458e8    # speed of light [m/s]
    
        T_d_z = T0 * (1 + z) ** alpha_cib
    
        x = -(3. + beta_cib + gamma_cib) * jnp.exp(-(3. + beta_cib + gamma_cib))
        # nu0 in GHz
        nu0_GHz = 1e-9 * k_B * T_d_z / h * (3. + beta_cib + gamma_cib + lambertw(x))
        # convert all nu, nu0 to Hz for Planck
        nu_Hz   = nu * 1e9      # If input is GHz!
        nu0_Hz  = nu0_GHz * 1e9
    
        def B_nu(nu_Hz, T):
            return (2 * h * nu_Hz ** 3 / c ** 2) / (jnp.exp(h * nu_Hz / (k_B * T)) - 1)
    
        
        Theta = jnp.where(
            nu_Hz >= nu0_Hz,
            (nu_Hz / nu0_Hz) ** (-gamma_cib),
            (nu_Hz / nu0_Hz) ** beta_cib * (B_nu(nu_Hz, T_d_z) / B_nu(nu0_Hz, T_d_z))
        )
        
        return Theta

        

    def l_gal(self, z, m, nu, params=None):
        params = merge_with_defaults(params)
    
        L0 = params["L0_cib"]
        
        
        # Note that Theta takes nu*(1+z) for SED instead of nu
        Phi = self.phi(z, m, nu, params)
        Theta = self.theta(z, m, nu, params)  
        Sigma = self.sigma(z, m, nu, params)

        
        return L0 * Phi * Sigma * Theta
        


    def l_sat(self, z, m, nu, params=None):
        params = merge_with_defaults(params)
        
        # Use a small fraction of host mass or a fixed value for min subhalo mass
        Ms_min = 1e5  # or Ms_min = 1e-4 * M_host - should eventually be a free parameter
        ngrid = 100   # Reasonable default for integration grid
    
        Ms_grid = jnp.logspace(jnp.log10(Ms_min), jnp.log10(m), ngrid)
        dlnMs = jnp.log(Ms_grid[1]/Ms_grid[0])
    
        # Subhalo mass function per dlnMs
        dN_dlnMs = self.subhalo_mass_function.dndlnmu(m, Ms_grid)
    
        # Galaxy luminosity for each subhalo mass
        L_gal = self.l_gal(z, Ms_grid, nu, params=params)
    
        # Integrate over ln(Ms)
        integrand = dN_dlnMs * L_gal
        L_sat = jnp.sum(integrand * dlnMs)
        return L_sat
        

    def l_cen(self, z, m, nu, params=None):
         params = merge_with_defaults(params)

         M_min = params["M_min_cib"]
         N_cen = jnp.where(m > M_min, 1.0, 0.0)

         # Galaxy luminosity for each subhalo mass
         L_gal = self.l_gal(z, m, nu, params=params)
         L_cen = N_cen * L_gal
         return L_cen

    def jbar_nu(self, z, m, nu, params=None):
        """
        Compute j̄_ν(z) = ∫dlnM (dn/dlnM) * (1/4π) * (Lc + Ls)
        
       
        Returns
        -------
        float
            j̄_ν(z) in appropriate units
        """
        params = merge_with_defaults(params)
        
        # Rest-frame frequency
        nu_rest = nu * (1 + z)
        
        # Get Lc and Ls at each mass (no NFW profile for monopole)
        Lc = jax.vmap(lambda m_i: self.l_cen(z, m_i, nu_rest, params=params))(m)
        Ls = jax.vmap(lambda m_i: self.l_sat(z, m_i, nu_rest, params=params))(m)
        
        # Halo mass function
        dndlnm = self.halo_model.halo_mass_function(z, m, params=params)
        
        # Integrand: (dn/dlnM) * (1/4π) * (Lc + Ls)
        integrand = dndlnm * (1.0 / (4.0 * jnp.pi)) * (Lc + Ls)
        
        # Integrate over ln(M)
        logm = jnp.log(m)
        jbar = jnp.trapezoid(integrand, x=logm)
        
        return jbar


    def cib_monopole(self, nu, z=jnp.geomspace(0.005, 3.0, 100), m=jnp.geomspace(1e10, 1e15, 100), params=None):
        """
        Compute the CIB monopole I_ν = ∫dz j̄_ν(z) * h² * (c/H)
        
        Returns
        -------
        float
            I_ν in Jy/sr
        """
        params = merge_with_defaults(params)
        
        
        h = params["H0"] / 100.0
        c_km_s = 299792.458  # speed of light in km/s
        
        # Compute j̄_ν(z) at each redshift
        jbar_grid = jax.vmap(lambda z_i: self.jbar_nu(z_i, m, nu, params=params))(z)
        
        # H(z) in km/s/Mpc
        Hz_grid = self.halo_model.emulator.hubble_parameter(z, params=params) * c_km_s  # Convert from 1/Mpc to km/s/Mpc
        
        # dχ/dz = c/H(z) in Mpc
        dchi_dz = c_km_s / Hz_grid
        
        # Integrand: j̄_ν * h² * (c/H)
        integrand = jbar_grid * h**2 * dchi_dz
        
        # Integrate over z
        I_nu = jnp.trapezoid(integrand, x=z)
        
        return I_nu



    def get_u_ell(self, z, m, moment=1, params=None):
        """ 
        Compute either the first or second moment of the CIB tracer u_ell.
        For CIB:, 
            First moment:     W_I_nu / jnu_bar * Lc + Ls * u_ell_m
            Second moment:     W_I_nu^2 / jnu_nu^2 * [Ls^2 * u_ell_m^2 + 2 * Ls * Lc * u_ell_m]
        You cannot simply take u_ell_g**2.

        Note that  W_I_nu = a(z) * jnu_bar, so  W_I_nu / jnu_bar = a(z)
        """

        params = merge_with_defaults(params)
        cparams = self.halo_model.emulator.get_all_cosmo_params(params)
        
        h = params["H0"]/100
        chi = self.halo_model.emulator.angular_diameter_distance(z, params=params) * (1 + z) * h 
        nu_rest = self.nu * (1 + z)

        s_nu_factor = jnp.sqrt(1 + z) / ((1 + z) * chi**2) 
        
        Ls = self.l_sat(z, m, nu_rest, params=params) 
        Lc = self.l_cen(z, m, nu_rest, params=params) 
        

        # Compute u_m_ell from BaseTracer
        ell, u_m = self.u_ell_analytic(z, m, params=params)

        rho_mean_0 = cparams["Rho_crit_0"] * cparams["Omega0_m"] 
        m_over_rho_mean = (m / rho_mean_0)[:, None]  # shape (N_m, 1)
        m_over_rho_mean = jnp.broadcast_to(m_over_rho_mean, u_m.shape)

        Hz = self.halo_model.emulator.hubble_parameter(z, params=params)

        #u_m *= m_over_rho_mean
    
        moment_funcs = [
            lambda _: 1 / h**2      / (4*jnp.pi)          * (Lc + Ls * u_m)                           * s_nu_factor        ,
            lambda _: 1 / h**4      / (4*jnp.pi)**2       * (Ls**2 * u_m**2 + 2 * Ls * Lc * u_m)      * s_nu_factor**2     ,
        ]


        #moment_funcs = [
        #    lambda _: 1 / h**4      / (4*jnp.pi)          * (Lc + Ls * u_m)                                                ,
        #    lambda _: 1 / h**8      / (4*jnp.pi)**2       * (Ls**2 * u_m**2 + 2 * Ls * Lc * u_m)                           ,
        #]
    
        u_ell = jax.lax.switch(moment - 1, moment_funcs, None)
    
        return ell, u_ell

    
