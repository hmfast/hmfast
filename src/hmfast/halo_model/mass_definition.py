import jax
import jax.numpy as jnp
from functools import partial

from hmfast.emulator import Emulator
from hmfast.defaults import merge_with_defaults
from hmfast.utils import newton_root


class Delta:
    """
    A class that defines the mass definition for halo models.

    Attributes
    ----------
    delta : int or str
        Overdensity threshold (e.g., 200, 500, "vir").
    delta_ref : str
        Reference density type: "critical" or "mean".
    """

    def __init__(self, delta, delta_ref):

        self.emulator = emulator 
        self._delta = None
        self._delta_ref = None
        self.delta_ref = delta_ref
        self.delta = delta


   # Ensure that delta_ref is only ever critical or mean
    @property
    def delta_ref(self):
        return self._delta_ref
    
    @delta_ref.setter
    def delta_ref(self, value):
        value = str(value).lower()
        if value not in ("critical", "mean"):
            raise ValueError("delta_ref must be either 'critical' or 'mean'")
            
        # Prevent changing delta_ref if delta == "vir"
        if getattr(self, "_delta", None) == "vir" and value != "critical":
            raise ValueError("'vir' is only allowed with 'critical' delta_ref")
        self._delta_ref = value

        
    @property
    def delta(self):
        return self._delta

        
    @delta.setter
    def delta(self, value):
        if isinstance(value, str):
            value = value.lower()
            
        # If 'vir', delta_ref must be 'critical'
        if value == "vir":
            if getattr(self, "_delta_ref", None) != "critical":
                raise ValueError("'vir' is only allowed with 'critical' delta_ref")
            self._delta = value
            return

        # Otherwise, it must be numeric
        if isinstance(value, (int, float)):
            self._delta = value
            return

        raise ValueError("delta must be numeric or 'vir'")


    def delta_vir_to_crit(self, z, params=None):
        """
        Bryan & Norman (1998) virial overdensity for a flat universe.
        Returns Δ_vir relative to the critical density.
    
        Returns
        -------
        float or array
            Δ_vir(z) relative to rho_crit
        """
        omega_m = self.emulator.omega_m_z(z, params=params)
        x = omega_m - 1.0
    
        return 18.0 * jnp.pi**2 + 82.0 * x - 39.0 * x**2

    def _delta_numeric(self, z, params=None):
        """ 
        Always return numeric delta at redshift z
        in the native reference (self.delta_ref).
        """
        if self.delta == "vir":
            if self.delta_ref != "critical":
                raise ValueError("virial overdensity only defined w.r.t. critical density")
            return self.delta_vir_to_crit(z, params=params)
    
        return self.delta


    def convert_delta_ref(self, z, delta, from_ref='critical', to_ref='mean', params=None):
        """
        Convert overdensity between 'critical' and 'mean' definitions.
        
        Parameters
        ----------
        delta : float or array
        z : float or array
        from_ref, to_ref : {'critical', 'mean'}
        """
        if from_ref == to_ref:
            return jnp.full_like(z, delta)
            
        omega_m = self.emulator.omega_m_z(z, params=params)
        if from_ref == 'critical' and to_ref == 'mean':
            return delta / omega_m
        elif from_ref == 'mean' and to_ref == 'critical':
            return delta * omega_m
        else:
            raise ValueError("from_ref and to_ref must be 'critical' or 'mean'")

         
    def delta_conversion_function(self, z, m_new, m_old, delta_old, delta_new, c_old, params=None):
        """
        Vectorized version: works for scalar or array inputs for z and m_new/m_old.
        Returns F(m_new) = m_new / m_old - f_NFW(c_old) / f_NFW(c_old * r_new / r_old)
        """
        params = merge_with_defaults(params)
        
        r_old = self.r_delta(z, m_old, delta_old, params=params)
        r_new = self.r_delta(z, m_new, delta_new, params=params)
       
        
        def f_nfw(x):
            return jnp.log1p(x) - x / (1.0 + x)
        
        return m_old / m_new - f_nfw(c_old) / f_nfw(c_old * r_new / r_old)


    @partial(jax.jit, static_argnums=0)
    def convert_m_delta(self, z, m, delta_old, delta_new, c_old, x0=None, max_iter=20, params=None):
        """
        Solve for m_{Δ'} given m_old, delta_old, delta_new, c_old, and redshift.
        Fully vectorized: computes all combinations of z, m, and c_old.
        """
        params = merge_with_defaults(params)
        if x0 is None:
            x0 = m
    
        # Make sure 1D arrays
        z = jnp.atleast_1d(z)
        m = jnp.atleast_1d(m)
        c_old = jnp.atleast_1d(c_old)
    
        # Broadcast to common shape
        z, m, c_old, x0 = jnp.broadcast_arrays(z, m, c_old, x0)
    
        # Solve for a single set (scalar z, m, c_old, x0)
        def solve_single(z_i, m_i, c_i, x0_i):
            F = lambda m_new: self.delta_conversion_function(z_i, m_new, m_i, delta_old, delta_new, c_i, params=params)
            return newton_root(F, x0=x0_i, max_iter=max_iter)
    
        # Vectorize over all elements
        solve_vec = jax.vmap(solve_single)
        return solve_vec(z, m, c_old, x0)



    def r_delta(self, z, m, delta, params=None):
        """
        Compute the halo radius corresponding to a given mass and overdensity at redshift z.
    
        Parameters
        ----------
        z : float
            Redshift at which to compute the radius.
        m : float
            Halo mass enclosed within the overdensity radius, in the same units as used for rho_crit.
        delta : float
            Overdensity parameter relative to the critical density (e.g., 200 for M_200).
        
        params : dict, optional
            Dictionary of cosmological parameters to use when computing the critical density.
    
        Returns
        -------
        float
            Radius r_delta (e.g., R_200) within which the average density equals delta * rho_crit(z).
        """
        params = merge_with_defaults(params)
        #cparams = get_all_cosmo_params(params)

        # Define your reference density. Default is rho_crit
        rho_ref = self.emulator.critical_density(z, params=params)

        # If the user selects vir or rho_mean, correct for this
        delta = self._delta_numeric(z, params=params)
        if self.delta_ref == "mean":
            rho_ref *= self.emulator.omega_m_z(z, params=params)
            
        return (3.0 * m / (4.0 * jnp.pi * delta * rho_ref))**(1./3.)

