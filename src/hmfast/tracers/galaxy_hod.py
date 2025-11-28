import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from hmfast.base_tracer import BaseTracer, HankelTransform
from jax.scipy.special import erf, sici  # Use jax-enabled math for special functions
import numpy as np
_eps = 1e-30
dndz_data = jnp.array(np.loadtxt("../data/hmfast_data/normalised_dndz_cosmos_0.txt"))


class GalaxyHODTracer(BaseTracer):
    """
    Galaxy HOD tracer implementing central + satellite occupation and
    NFW satellites.

    Parameters
    ----------
    emulator : 
        Cosmological emulator used to compute cosmological quantities
    params : dict
        Dictionary of parameters. 
        Relevant keys:
          - M_min_HOD, sigma_log10M_HOD, M0_HOD, M1_prime_HOD, alpha_s_HOD
    """

    def __init__(self, emulator, halo_model, params):        
        self.params = params
        x_min, x_max, x_npoints = params['x_min'], params['x_max'], params['x_npoints']
        self.x_grid = jnp.logspace(jnp.log10(x_min), jnp.log10(x_max), x_npoints)
        self.hankel = HankelTransform(x_min=x_min, x_max=x_max, x_npoints=x_npoints, nu=0.5)

        self.emulator = emulator # cosmology emulator
        self.halo_model = halo_model

    # -------------------------
    # HOD functions (vectorized)
    # -------------------------
    def get_N_centrals(self, m, params = None):
        """Mean central occupation: shape = M.shape"""
        M_min = params["M_min_HOD"]
        sigma = params["sigma_log10M_HOD"]
        x = (jnp.log10(m) - jnp.log10(M_min)) / sigma
        return 0.5 * (1.0 + erf(x))

    def get_N_satellites(self, m, params = None):
        """Mean satellite occupation: shape = M.shape"""
        M0 = params["M0_HOD"]
        M1p = params["M1_prime_HOD"]
        alpha = params["alpha_s_HOD"]

        # power law only above M0 and use jnp.where to keep differentiability
        pow_term = jnp.maximum((m - M0) / M1p, 0.0)**alpha
        N_c = self.get_N_centrals(m, params = params)
        return  N_c * pow_term
        

    def get_ng_bar_at_z(self, z, params = None):
        """
        Compute comoving galaxy number density ng(z) = ∫ dlnM [dn/dlnM] [Nc+Ns].
        halo_model: HaloModel instance
        tracer: GalaxyHODTracer instance (provides HOD via helper funcs)
        z: scalar redshift
        params: parameter dict (used to construct the m_grid etc.)
        """
        # mass grid (same convention used in HaloModel.get_C_ell_*)
        m_grid = jnp.geomspace(params['M_min'], params['M_max'], params['M_npoints'])
        logm = jnp.log(m_grid)
        z = jnp.atleast_1d(z)

        Nc = self.get_N_centrals(m_grid, params=params)
        Ns = self.get_N_satellites(m_grid, params=params)
        Ntot = Nc + Ns
    
        def ng_bar_single(z_single):
            dndlnm = self.halo_model.mass_function(z_single, m_grid, params=params)  # shape (n_m,)
            integrand = dndlnm * Ntot
            return jnp.trapezoid(integrand, x=logm)
    
        # vectorize over z
        return jax.vmap(ng_bar_single)(z)

    

    def get_wg_at_z_alt(self, z, params=None):
        """
        Compute Wg(z) = [H(z) / c] * [phi'_g(z) / chi(z)^2] following Eq. (13)
        of the paper https://arxiv.org/pdf/2203.12583, 
        where phi'_g(z) = (1 / N_tot) * dN_g/dz and
        dN_g/dz = ng(z) * dV/dz/dOmega.

        """

        
        z_grid = jnp.geomspace(params['z_min'], params['z_max'], params['z_npoints'])


        # Compute ng(z) on the grid using the existing get_ng_bar_at_z
        ng_grid = self.get_ng_bar_at_z(z_grid, params=params)  # shape (n_z,)

        # Compute dNdz given that dN/dz/dOmega = ng(z) * dV/dz/dOmega
        dVdz_grid = self.emulator.get_dVdzdOmega_at_z(z_grid, params=params)  # shape (n_z,)
        dNdz = ng_grid * dVdz_grid

        # Total number N_tot = ∫ dN/dz dz (per steradian)
        N_tot = jnp.trapezoid(dNdz, x=z_grid)
        N_tot_safe = jnp.maximum(N_tot, _eps)

        # Normalized galaxy distribution phi'_g(z) = dN/dz / N_tot
        phi_prime = dNdz / N_tot_safe

        # Get H(z) and chi(z). No need to divide H by c since the emulator outputs 1/Mpc
        H_grid = self.emulator.get_hubble_at_z(z_grid, params=params)  # 1/Mpc
        chi_grid = self.emulator.get_angular_distance_at_z(z_grid, params=params) * (1.0 + z_grid)  # Mpc comov

        # Assemble Wg on the grid
        chi2_safe = jnp.maximum(chi_grid ** 2, _eps)
        Wg_grid = H_grid * (phi_prime / chi2_safe)

        # Interpolate Wg_grid to requested z
        zq = jnp.atleast_1d(jnp.array(z, dtype=jnp.float64))
        Wg_q = jnp.interp(zq, z_grid, Wg_grid, left=0.0, right=0.0)

        z = dndz_data[:, 0]          # first column: redshifts
        phi_prime = dndz_data[:, 1]  # second column: phi_prime
        #print("phi_prime", phi_prime)
        return Wg_q

   


    def get_wg_at_z(self, z, params=None):
        """
        Return Wg_grid at requested z.
        Uses pre-loaded dndz_data = [z, phi_prime].
        """
        zq = jnp.atleast_1d(jnp.array(z, dtype=jnp.float64))
    
        # Extract precomputed phi_prime
        z_data = dndz_data[:, 0]
        phi_prime_data = dndz_data[:, 1]
    
        # Interpolate phi_prime to requested z
        phi_prime_at_z = jnp.interp(zq, z_data, phi_prime_data, left=0.0, right=0.0)
    
        H_grid = self.emulator.get_hubble_at_z(zq, params=params)  # 1/Mpc
        chi_grid = self.emulator.get_angular_distance_at_z(zq, params=params) * (1.0 + zq)  # Mpc comov

        # Assemble Wg on the grid
        Wg_grid = H_grid * (phi_prime_at_z / chi_grid**2)
        return Wg_grid


    def c_Duffy2008(self, z, m, A=5.71, B=-0.084, C=-0.47, M_pivot=2e12):
        """
        Duffy et al. 2008 mass-concentration relation.
        
        Parameters:
            M : float or array, halo mass in Msun/h
            z : float or array, redshift
            A, B, C : fit parameters
            M_pivot : pivot mass (Msun/h)
        
        Returns:
            c : concentration
        """
        return A * (m / M_pivot)**B * (1 + z)**C



        

    def compute_u_m_ell(self, z, m, params = None):
        rparams = self.emulator.get_all_relevant_params(params)
        rho_m0 = rparams['Omega0_m'] * rparams["Rho_crit_0"] 
        c_200c = self.c_Duffy2008(z, m)
        B = params["B"] 

        m = jnp.atleast_1d(m) 
        ell_min = params.get("ell_min", 1e2)
        ell_max = params.get("ell_max", 3.5e3) 
        
        r_200c = self.emulator.get_r_delta_of_m_delta_at_z(200, m, z, params=params) / B**(1/3)

        chi = self.emulator.get_angular_distance_at_z(z, params=params) * (1.0 + z)
        ell_row = jnp.logspace(jnp.log10(ell_min), jnp.log10(ell_max), 100)

        ell = jnp.broadcast_to(ell_row[None, :], (m.shape[0], 100))           # (N_m, N_k)
        k = (ell_row + 0.5) / chi   # physical k
  
        
        k_mat = k[None, :]                            # (1, N_k)
        r_mat = r_200c[:, None]                       # (N_m, 1)
        c_mat = jnp.atleast_1d(c_200c)[:, None]       # (N_m, 1)
        
        lambda_val = 1
        
        q = k_mat * r_mat / c_mat            # (N_m, N_k)
        q_scaled = (1 + lambda_val * c_mat) * q

        f_nfw = lambda x: 1.0 / (jnp.log1p(x) - x/(1 + x))
        f_nfw_val = f_nfw(lambda_val * c_mat)

        
        Si_q, Ci_q = sici(q)
        Si_q_scaled, Ci_q_scaled = sici(q_scaled)
        #Ci_q, Ci_q_scaled = -1 * Ci_q, -1 * Ci_q_scaled  # We need to flip the sign of Ci since Ci_JAX(x) = - Ci_needed(x)

        
        u_ell_m = (m[:, None] / rho_m0) * ( jnp.cos(q) * (Ci_q_scaled - Ci_q) +
                                            jnp.sin(q) * (Si_q_scaled - Si_q) - 
                                            jnp.sin(lambda_val * c_mat * q)/(q_scaled) ) * f_nfw_val
   

        return ell, u_ell_m


    def compute_u_ell(self, z, m, params = None):
        Nc = self.get_N_centrals(m, params=params) 
        Ns = self.get_N_satellites(m, params=params)

        ng_bar = self.get_ng_bar_at_z(z, params = params)
        W_g = self.get_wg_at_z(z, params=params)
        
        pref = W_g / ng_bar 

        ell, u_m_ell = self.compute_u_m_ell(z, m, params=params)
        return ell, pref[:, None] * (Nc[:, None] + Ns[:, None] * u_m_ell)


    def compute_u_ell_squared(self, z, m, params = None):
        """ 
        For galaxy HOD:, 
           〈|u_ell^g(M, z)|^2〉= W_g(z) / ng_bar(z)^2 * [Ns^2 u_ell^g(M, z)^2 + 2Ns u_ell^g(M, z)]
        You cannot simply take [u_ell^g(M, z)]^2.
        """

        Ns = self.get_N_satellites(m, params=params)
        ng_bar = self.get_ng_bar_at_z(z, params=params)
        W_g = self.get_wg_at_z(z, params=params)
        ell, u_m_ell = self.compute_u_m_ell(z, m, params=params)
        pref = W_g / ng_bar**2
        
        u_ell_squared =  pref[:, None] * ( Ns[:, None]**2 * u_m_ell**2 + 2 * Ns[:, None] * u_m_ell )

        return ell, u_ell_squared



