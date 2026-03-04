import jax
import jax.numpy as jnp
from functools import partial


class T08HaloMass:
    """
    Tinker et al. (2008) halo mass function.

    Computes the differential mass function dn/dlnσ for given
    variance σ(R) over a range of redshifts.

    Parameters
    ----------
    sigmas : jnp.ndarray
        Variance of the linear density field σ(R, z), shape (n_R, n_z) or (n_R,)
    z : float or jnp.ndarray
        Redshift(s) corresponding to sigmas
    delta_mean : float or jnp.ndarray
        Halo overdensity Δ (e.g., 200, 500, 1600). Can be scalar or shape (n_z,)

    Returns
    -------
    f_sigma : jnp.ndarray
        Halo mass function values, shape matching sigmas
        (dn/dlnσ) in units consistent with Tinker et al. (2008)
    """

    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def f_sigma(self, sigmas, z, delta_mean):
        # Convert delta_mean to log scale
        delta_mean = jnp.log10(delta_mean)
        
        # Define parameters as JAX arrays
        delta_mean_tab = jnp.log10(jnp.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200]))
        A_tab = jnp.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
        aa_tab = jnp.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
        b_tab = jnp.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
        c_tab = jnp.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
    
        # Linear interpolation using jnp.interp
        Ap = jnp.interp(delta_mean, delta_mean_tab, A_tab) * (1 + z) ** -0.14
        a = jnp.interp(delta_mean, delta_mean_tab, aa_tab) * (1 + z) ** -0.06
        b = jnp.interp(delta_mean, delta_mean_tab, b_tab) * (1 + z) ** -jnp.power(10, -jnp.power(0.75 / jnp.log10(jnp.power(10, delta_mean) / 75), 1.2))
        c = jnp.interp(delta_mean, delta_mean_tab, c_tab)
        
        # Calculate final result f(σ)
        f_sigma = 0.5 * Ap[:, None] * (jnp.power(sigmas / b[:, None], -a[:, None]) + 1) * jnp.exp(-c[:, None] / sigmas**2)
        return f_sigma




class T10HaloMass:
    """
    Tinker et al. (2010) halo mass function f(ν, z).
    """

    def __init__(self):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def f_sigma(self, sigmas, z, delta_mean):
        """
        Tinker et al. (2010) halo mass function f(nu, z)
        nu : peak-height (delta_c / sigma)
        z  : redshift
        Returns f(nu,z) shape (n_z, n_nu)
        """
        delta_mean = jnp.log10(delta_mean)
        delta_c = 1.686 
        log_nu = 2.0 * jnp.log(delta_c) - 2.0 * jnp.log(sigmas)
        nu = jnp.exp(log_nu)
        
        # Base parameters
        alpha0, beta0, gamma0, eta0, phi0 = 0.368, 0.589, 0.864, -0.243, -0.729
        # Redshift exponents
        alpha_z, beta_z, gamma_z, eta_z, phi_z = 0.0, 0.2, -0.01, 0.27, -0.08

        # Compute z-dependent parameters
        alpha = alpha0 * (1 + z)**alpha_z
        beta  = beta0  * (1 + z)**beta_z
        gamma = gamma0 * (1 + z)**gamma_z
        eta   = eta0   * (1 + z)**eta_z
        phi   = phi0   * (1 + z)**phi_z

        # Reshape for broadcasting
        alpha = alpha[:, None]
        beta  = beta[:, None]
        gamma = gamma[:, None]
        eta   = eta[:, None]
        phi   = phi[:, None]

        beta_term = (beta ** 2 * nu) ** (-phi)  # (beta^2 * nu)^(-phi)
        eta_term  = nu ** eta
        exp_term  = jnp.exp(-gamma * nu / 2)


        f_nu = 0.5 * alpha * (1 + beta_term) * eta_term * exp_term * jnp.sqrt(nu)
        return f_nu




class T10HaloBias:
    """
    Tinker et al. (2010) large-scale linear bias, JAX-friendly.

    Parameters
    ----------
    sigmas : jnp.ndarray
        sigma(R,z) or sigma(M,z), shape (nM, n_z)
    z : scalar or array_like
        Redshift(s) (kept for API compatibility)
    delta_mean : scalar or array_like
        Halo overdensity Δ, shape (n_z,) or scalar

    Returns
    -------
    b_nu : jnp.ndarray
        Halo bias, shape same as sigmas
    """

    def __init__(self):
        pass


    @partial(jax.jit, static_argnums=(0,))
    def b1_nu(self, sigmas, z, delta_mean):
        y = jnp.log10(delta_mean)
        delta_c = 1.686  # the critical overdensity (slightly redshift-dependent in LCDM), so this is approximate
        
        # Tinker (2010) parameters
        A  = jnp.array(1.0 + 0.24 * y * jnp.exp(-(4.0 / y) ** 4))
        a  = jnp.array(0.44 * y - 0.88)
        B  = jnp.array(0.183)
        b_ = jnp.array(1.5)
        C  = jnp.array((0.019 + 0.107 * y + 0.19 * jnp.exp(-(4.0 / y) ** 4)))
        c  = jnp.array(2.4)
    
        nu = delta_c / sigmas
        nu_a = jnp.power(nu, a)
        first = A * (nu_a / (nu_a + delta_c ** a))
        b_nu = 1.0 - first + B * jnp.power(nu, b_) + C * jnp.power(nu, c)
    
        return b_nu


    @partial(jax.jit, static_argnums=(0,))
    def b2_nu(self, sigmas, z, delta_mean):

        delta_c =  1.686
        nu = (delta_c / sigmas)**2

        z = jnp.atleast_1d(z)
        
        # Base parameters followed by redshift exponents
        alpha0, beta0, gamma0, eta0, phi0 = 0.368, 0.589, 0.864, -0.243, -0.729
        alpha_z, beta_z, gamma_z, eta_z, phi_z = 0.0, 0.2, -0.01, 0.27, -0.08

        # Compute z-dependent parameters
        alpha = alpha0 * (1 + z)**alpha_z
        beta  = beta0  * (1 + z)**beta_z
        gamma = gamma0 * (1 + z)**gamma_z
        eta   = eta0   * (1 + z)**eta_z
        phi   = phi0   * (1 + z)**phi_z


        a = -phi
        b = beta**2
        c = gamma
        d = eta + 0.5

        a2 = -17/21
        

        eps1 = (c * nu - 2 * d) / delta_c
        eps2 = (c * nu * (c * nu - 4 * d - 1) + 2 * d * (2 * d - 1)) /  delta_c**2
        
        E1 = - 2 * a / (delta_c * ((b * nu)**(-a) + 1))
        E2 = E1  * (-2 * a + 2 * c * nu - 4 * d + 1) / delta_c

        b2_nu = 2 * (1 + a2) * (eps1 + E1) + eps2 + E2

        return b2_nu


class TW10SubHaloMass:
    """
    Tinker & Wetzel (2010) subhalo mass function (Eq. 2).

    Parameters
    ----------
    M_host : float or array_like
        Host halo mass [Msun]
    M_sub : float or array_like
        Subhalo mass [Msun]

    Returns
    -------
    dN_dlnmu : float or array_like
        Number of subhalos per host per dln(mu)
    """
    def __init__(self):
        pass
    
    def dndlnmu(self, M_host, M_sub):
        mu = M_sub / M_host
        dN_dlnmu = 0.30 * mu ** (-0.7) * jnp.exp(-9.9 * mu ** 2.5)
        return dN_dlnmu



class D08Concentration:
    """
    Duffy et al. (2008) mass-concentration relation.
    A, B, C are fit parameters, and M_pivot is the pivot mass (Msun/h)
    """
    def __init__(self):
        pass


    def c_delta(self, halo_model, z, m, params):
        # Probably a prettier way of doing this
        if halo_model.delta == 200 and halo_model.delta_ref == "critical":
            A, B, C, M_pivot = 5.71, -0.084, -0.47, 2e12
        elif halo_model.delta == 200 and halo_model.delta_ref == "mean":
            A, B, C, M_pivot = 10.14, -0.081, -1.01, 2e12
        elif halo_model.delta == "vir":
            A, B, C, M_pivot = 7.85, -0.081, -0.71, 2e12
        else:
            raise ValueError("The c-M relation c_D08 is incompatible with the chosen definiton of delta. You must select from the following: '200c', '200m', 'vir'.")
    
        return A * (m / M_pivot)**B * (1 + z)**C




class B13Concentration:
    """
    Bhattacharya et al. (2013) mass-concentration relation for c200_c.
    Obtained from Table 2, https://arxiv.org/pdf/1112.5479
    D here is the growth factor D(z).
    """


    def __init__(self):
        pass


    def c_delta(self, halo_model, z, m, params):
        # Use the nu as defined in the B13 paper and pivot mass in Msun/h
        D = halo_model.emulator.growth_factor(z, params=params)
    
        # Probably a prettier way of doing this
        if halo_model.delta == 200 and halo_model.delta_ref == "critical":
            A, B, C = 5.9, 0.54, -0.35
        elif halo_model.delta == 200 and halo_model.delta_ref == "mean":
            A, B, C = 9.0, 1.15, -0.29
        elif halo_model.delta == "vir":
            A, B, C = 7.7, 0.9, -0.29
        else:
            raise ValueError("The c-M relation c_B13 is incompatible with the chosen definiton of delta. You must select from the following: '200c', 200m', 'vir'.")
    
        
        nu = (1.12 * (m / 5e13)**0.3 + 0.53) / D
        c_delta = A * D**B * nu**C
        return c_delta


class ConstantConcentration:
    """
    Constant concentration-mass relation, with the value of c_delta being specified in the parameters.
    """
    def __init__(self, c):
        self.c = c
        pass

    def c_delta(self, halo_model, z, m, params):
        return jnp.full_like(m, self.c)




class SC14Concentration:
    """
    Sanchez-Conde & Prada (2014) concentration-mass relation for c200_c.
    Coefficients are found below Equation 1, https://arxiv.org/pdf/1312.1729
    """
    def __init__(self):
        pass

    
    def c_delta(self, halo_model, z, m, params=None):
        # Coefficients from Eq. 1
        if halo_model.delta == 200 and halo_model.delta_ref == "critical":
            c_array = jnp.array([37.5153, -1.5093, 1.636e-2, 3.66e-4, -2.89237e-5, 5.32e-7])
            logM = jnp.log10(m)
            powers = jnp.array([logM**i for i in range(6)])
            poly = jnp.sum(c_array[:, None] * powers, axis=0)
            c_delta = poly * (1 + z) ** -1
    
        else: 
            raise ValueError("The c-M relation c_SC14 is incompatible with the chosen definiton of delta. You must select from the following: '200c'.")
        return c_delta
        