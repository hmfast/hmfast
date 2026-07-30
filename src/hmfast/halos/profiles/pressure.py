import os
import numpy as np
import jax
import jax.numpy as jnp
import mcfit
from scipy.special import ive
from hmfast.download import get_default_data_path
from hmfast.utils import Const
from hmfast.halos.mass_definition import MassDefinition, convert_m_delta
from hmfast.halos.profiles import HaloProfile, HankelTransform



class PressureProfile(HaloProfile):
    @jax.jit
    def u_k(self, halo_model, k, m, z):
        """
        Compute the projected Fourier-space pressure profile for halo-model calculations.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology and halo-radius relation.
        k : float or jnp.ndarray
            Comoving wavenumber(s) in :math:`\\mathrm{Mpc}^{-1}`.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Transformed profile with shape :math:`(N_k, N_m, N_z)`.
        """
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        B = getattr(self, "B", 1.0)

        r_delta_native = halo_model.mass_definition.r_delta(halo_model.cosmology, m, z)
        r_delta = r_delta_native / B**(1 / 3)
        r = self.x[:, None, None] * r_delta_native[None, :, :] * (1.0 + z[None, None, :])
        d_A = jnp.atleast_1d(halo_model.cosmology.angular_diameter_distance(z))
        ell_delta = d_A[None, :] / r_delta  # (Nm, Nz)
        
        mpc_to_m = Const._Mpc_over_m_
        prefactor = (1 + z)[None, :] * 4 * jnp.pi * r_delta * mpc_to_m / (ell_delta**2)  # (Nm, Nz)
        
        # Target ell grid for interpolation: (Nk, Nz)
        chi = d_A * (1 + z)
        ell_target = k[:, None] * chi[None, :] - 0.5 
        
        # Get native Hankel transform outputs, which may not align with the k from this function's input
        k_native, u_k_native = self._u_k_hankel(halo_model, self.x, r, m, z)
        
        # Calculate native u_ell and the native ell grid
        u_ell_native = u_k_native * jnp.sqrt(jnp.pi / (2 * k_native[:, None, None])) 
        ell_native = k_native[:, None, None] * ell_delta[None, :, :] # (Nk_native, Nm, Nz)
        
        # Apply prefactor
        u_ell_base = prefactor[None, :, :] * u_ell_native # (Nk_native, Nm, Nz)
        u_ell_val = u_ell_base
    
        # Interpolate over the native k-axis (axis 0) for every combination of m and z    
        def interp_at_z(ell_t, ell_n, u_n):
            return jnp.interp(ell_t, ell_n, u_n)
       
        vmap_interp = jax.vmap(
            jax.vmap(interp_at_z, in_axes=(None, 1, 1), out_axes=1), 
            in_axes=(1, 2, 2), out_axes=2
        )
        
        # Resulting shape: (Nk, Nm, Nz)
        u_ell_interp = vmap_interp(ell_target, ell_native, u_ell_val)
        
        return u_ell_interp






class GNFWPressureProfile(PressureProfile):
    """
    Electron pressure profile from `Nagai, Kravtsov & Vikhlinin (2007) <https://ui.adsabs.harvard.edu/abs/2007ApJ...668....1N/abstract>`_.

    The profile is evaluated as a function of the comoving radius
    :math:`r`, but its normalization and shape are defined using the native
    :math:`500c` calibration mass and radius:

    .. math::

        P_e(r, M, z) = P_{500c} \\, P_0
        \\left(c_{500} x_{500c}\\right)^{-\\gamma}
        \\left[1 + \\left(c_{500} x_{500c}\\right)^\\alpha\\right]^{(\\gamma-\\beta)/\\alpha}
        \\tag{1}

    where :math:`x_{500c} = r / r_{500c}` and :math:`r_{500c}` has the same
    units as :math:`r`, and

    .. math::

        P_{500c} =
        1.65
        \\left(\\frac{h}{0.7}\\right)^2
        E(z)^{8/3}
        \\left(\\frac{M_{500c} / B}{0.7 \\times 3 \\times 10^{14} \\, M_\\odot}\\right)^{2/3 + 0.12}
        \\left(\\frac{0.7}{h}\\right)^{3/2}
        \\tag{2}

    with :math:`E(z) = H(z) / H_0`. In the implementation, the input halo mass
    is first converted from the halo model's mass definition to
    :math:`M_{500c}`.

    The projected Fourier-space pressure profile is evaluated as

    .. math::

        u_\\ell(\\ell, M, z) =
        \\frac{4 \\pi (1+z) r_\\Delta}{\\ell_\\Delta^2}
        \\int dx \\, x^2 \\, P_e(x, M, z)
        \\, \\frac{\\sin\\!\\left[(\\ell / \\ell_\\Delta) x\\right]}
        {(\\ell / \\ell_\\Delta) x}
        \\tag{3}

    where :math:`\\ell_\\Delta(M, z) = d_A(z) / r_\\Delta(M, z)` and
    :math:`\\chi(z) = (1+z) d_A(z)`.

    Attributes
    ----------
    x : jnp.ndarray
        Dimensionless radial grid :math:`x = r / r_\\Delta` used to tabulate the profile and define the Hankel transform, with :math:`r_\\Delta` expressed in the same units as :math:`r`.
    P0 : float
        Dimensionless gNFW normalization :math:`P_0`.
    c500 : float
        Concentration parameter :math:`c_{500}` of the :math:`500c` pressure profile.
    alpha : float
        Intermediate-slope parameter :math:`\\alpha` of the gNFW profile.
    beta : float
        Outer-slope parameter :math:`\\beta` of the gNFW profile.
    gamma : float
        Inner-slope parameter :math:`\\gamma` of the gNFW profile.
    B : float
        Hydrostatic mass bias factor :math:`B` used in the :math:`M_{500c}` normalization.
    """
    
    def __init__(self, x=None, P0=8.130, c500=1.156, alpha=1.0620, beta=5.4807, gamma=0.3292, B=1.4):

        self.P0 = P0
        self.c500 = c500
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.B = B

        if x is not None:
            self.x = x
        else:
            # Match tszpower/profiles.py:14 (_X_GRID = logspace(-6, 6, 1024))
            # so the Hankel transform of GNFW captures the large-x tail. The
            # GNFW decays as r^{-beta} (beta~5.5) so beyond x~100 contributions
            # are < 1e-10 of central; the wide range is needed for the Hankel
            # to be numerically well-conditioned, not for the physical integrand.
            self.x = np.logspace(np.log10(1e-6), np.log10(1e6), 1024)


    @property
    def x(self):
       return self._x

    @x.setter
    def x(self, value):
        """
        Whenever x is modified, immediately rebuild the hankel transform object.
        Stores x as numpy so it is safe to use as JAX PyTree aux_data.
        """
        self._x = np.asarray(value)
        self._hankel = HankelTransform(jnp.asarray(self._x), nu=0.5)


    def _tree_flatten(self):
        leaves = (self.P0, self.c500, self.alpha, self.beta, self.gamma, self.B)
        # Store _x as tuple for hashable aux_data with proper __eq__
        aux_data = (tuple(self._x.tolist()), self._hankel)
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        x_tuple, hankel = aux_data
        obj = cls.__new__(cls)
        obj.P0, obj.c500, obj.alpha, obj.beta, obj.gamma, obj.B = leaves
        obj._x = np.array(x_tuple)
        obj._hankel = hankel
        return obj

    def update(self, P0=None, c500=None, alpha=None, beta=None, gamma=None, B=None):
        """
        Return a new profile instance with updated GNFW pressure profile parameters.

        Parameters
        ----------
        P0 : float, optional
        c500 : float, optional
        alpha : float, optional
        beta : float, optional
        gamma : float, optional
        B : float, optional

        Returns
        -------
        GNFWPressureProfile
            New profile instance with updated parameters.
        """
        leaves, treedef = self._tree_flatten()

        new_leaves = (
            P0 if P0 is not None else self.P0,
            c500 if c500 is not None else self.c500,
            alpha if alpha is not None else self.alpha,
            beta if beta is not None else self.beta,
            gamma if gamma is not None else self.gamma,
            B if B is not None else self.B,
        )

        return self._tree_unflatten(treedef, new_leaves)

    @jax.jit
    def u_r(self, halo_model, r, m, z):
        """
        Compute the electron-pressure profile.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, mass-definition conversion, and halo
            radius.
        r : float or jnp.ndarray
            Comoving radius or radii in :math:`\\mathrm{Mpc}`.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Electron pressure profile with shape :math:`(N_r, N_m, N_z)`.
        """
        H0 = halo_model.cosmology.H0
        P0, c500, alpha, beta, gamma, B = self.P0, self.c500, self.alpha, self.beta, self.gamma, self.B
        r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m), jnp.atleast_1d(z)
        h = H0 / 100.0
    
        # Convert input mass to M500c for normalization, since this profile was calibrated for 500c
        mass_def_old = halo_model.mass_definition
        mass_def_500c = MassDefinition(500, "critical")
        c_old = halo_model.concentration.c_delta(halo_model, m, z)
        m500c = convert_m_delta(halo_model.cosmology, m, z, mass_def_old, mass_def_500c, c_old=c_old)
    
        r_500c = mass_def_500c.r_delta(halo_model.cosmology, m500c, z)  # (Nm, Nz)

        # Convert the comoving radius to the calibrated physical 500c coordinate.
        x_500c = r[:, None, None] / ((1.0 + z[None, None, :]) * r_500c[None, :, :])  # (Nr, Nm, Nz)

        # Compute normalization P_500c (with hydrostatic bias)
        h = H0 / 100.0
        H = halo_model.cosmology.hubble_parameter(z)  # (Nz,)
        H = jnp.atleast_1d(H)[None, None, :]  # (1, 1, Nz)
        m500c_tilde = (m500c * h / B)[None]  # (1, Nm, Nz)
        P_500c = (1.65 * (h / 0.7) ** 2 * (H / H0) ** (8 / 3) * (m500c_tilde / (0.7 * 3e14)) ** (2 / 3 + 0.12) * (0.7 / h) ** 1.5)  # (1, Nm, Nz)
    
        # GNFW profile
        scaled_x = c500 * x_500c  # (Nr, Nm, Nz)
        Pe = P_500c * P0 * scaled_x ** (-gamma) * (1 + scaled_x ** alpha) ** ((gamma - beta) / alpha)
    
        return Pe  # shape: (Nr, Nm, Nz)

jax.tree_util.register_pytree_node(
    GNFWPressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: GNFWPressureProfile._tree_unflatten(aux_data, children)
)


class ParametricGNFWPressureProfile(PressureProfile):
    """
    GNFW pressure profile with a parametric Compton-:math:`y_0` amplitude.

    The radial shape and Arnaud :math:`P_{500c}` normalization are unchanged
    relative to :class:`GNFWPressureProfile`; the profile is multiplied by
    a mass- and redshift-dependent ratio

    .. math::

        \\mathrm{ratio}(M, z) = \\frac{y_0^{\\rm param}(M, z)}{y_0^{\\rm orig}(M, z)}

    where, matching ``tszpower.parametric_profile.compute_y0_parametric``,

    .. math::

        y_0^{\\rm param}(M, z) = 10^{A_{\\rm SZ}}
        \\left(\\frac{M_{500c} h / B}{0.7 \\times 3 \\times 10^{14} \\, M_\\odot}\\right)^{\\alpha_{\\rm SZ}}
        E(z)^2 \\left(\\frac{h}{0.7}\\right)^{-1/2}

    and :math:`y_0^{\\rm orig}` is the central Compton-:math:`y` of the Arnaud
    GNFW profile evaluated at the same :math:`(M, z)`:

    .. math::

        y_0^{\\rm orig}(M, z) = 2 \\frac{\\sigma_T}{m_e c^2}
        \\, P_0 \\, P_{500c}^{\\rm Arnaud}(M, z) \\, r_{500c}(M, z)
        \\, \\mathcal{I}_{\\rm shape}

    with the dimensionless GNFW shape integral
    :math:`\\mathcal{I}_{\\rm shape} = 0.470502095` (matching tszpower).
    When :math:`A_{\\rm SZ}, \\alpha_{\\rm SZ}` give :math:`y_0^{\\rm param}
    = y_0^{\\rm Arnaud}`, ``ratio = 1`` and the profile reduces exactly to
    :class:`GNFWPressureProfile`.

    Attributes
    ----------
    x : jnp.ndarray
        Dimensionless radial grid :math:`x = r / r_\\Delta`.
    A_SZ : float
        Log10 amplitude of the parametric normalization.
    alpha_SZ : float
        Mass-scaling exponent of the parametric normalization.
    P0, c500, alpha, beta, gamma : float
        gNFW shape parameters (same as :class:`GNFWPressureProfile`).
    B : float
        Hydrostatic mass bias factor used in the :math:`M_{500c}` normalization.
    """

    def __init__(self, x=None, A_SZ=-4.97, alpha_SZ=0.7867,
                 P0=8.130, c500=1.156, alpha=1.0620, beta=5.4807, gamma=0.3292,
                 B=1.4):

        self.A_SZ = A_SZ
        self.alpha_SZ = alpha_SZ
        self.P0 = P0
        self.c500 = c500
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.B = B

        if x is not None:
            self.x = x
        else:
            # Match tszpower/profiles.py:14 (_X_GRID = logspace(-6, 6, 1024))
            # so the Hankel transform of GNFW captures the large-x tail. The
            # GNFW decays as r^{-beta} (beta~5.5) so beyond x~100 contributions
            # are < 1e-10 of central; the wide range is needed for the Hankel
            # to be numerically well-conditioned, not for the physical integrand.
            self.x = np.logspace(np.log10(1e-6), np.log10(1e6), 1024)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.asarray(value)
        self._hankel = HankelTransform(jnp.asarray(self._x), nu=0.5)

    def _tree_flatten(self):
        leaves = (self.A_SZ, self.alpha_SZ,
                  self.P0, self.c500, self.alpha, self.beta, self.gamma, self.B)
        aux_data = (tuple(self._x.tolist()), self._hankel)
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        x_tuple, hankel = aux_data
        obj = cls.__new__(cls)
        (obj.A_SZ, obj.alpha_SZ,
         obj.P0, obj.c500, obj.alpha, obj.beta, obj.gamma, obj.B) = leaves
        obj._x = np.array(x_tuple)
        obj._hankel = hankel
        return obj

    def update(self, A_SZ=None, alpha_SZ=None,
               P0=None, c500=None, alpha=None, beta=None, gamma=None, B=None):
        """
        Return a new profile instance with updated parameters.

        Parameters
        ----------
        A_SZ, alpha_SZ : float, optional
            Parametric amplitude parameters.
        P0, c500, alpha, beta, gamma, B : float, optional
            gNFW shape and bias parameters.

        Returns
        -------
        ParametricGNFWPressureProfile
            New profile instance with updated parameters.
        """
        leaves, treedef = self._tree_flatten()

        new_leaves = (
            A_SZ if A_SZ is not None else self.A_SZ,
            alpha_SZ if alpha_SZ is not None else self.alpha_SZ,
            P0 if P0 is not None else self.P0,
            c500 if c500 is not None else self.c500,
            alpha if alpha is not None else self.alpha,
            beta if beta is not None else self.beta,
            gamma if gamma is not None else self.gamma,
            B if B is not None else self.B,
        )

        return self._tree_unflatten(treedef, new_leaves)

    @jax.jit
    def u_r(self, halo_model, r, m, z):
        """
        Compute the electron-pressure profile with a parametric amplitude.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, mass-definition conversion, and
            halo radius.
        r : float or jnp.ndarray
            Comoving radius or radii in :math:`\\mathrm{Mpc}`.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Electron pressure profile with shape :math:`(N_r, N_m, N_z)`.
        """
        H0 = halo_model.cosmology.H0
        A_SZ, alpha_SZ = self.A_SZ, self.alpha_SZ
        P0, c500, alpha, beta, gamma, B = (
            self.P0, self.c500, self.alpha, self.beta, self.gamma, self.B
        )
        r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m), jnp.atleast_1d(z)
        h = H0 / 100.0

        # Convert input mass to M500c
        mass_def_old = halo_model.mass_definition
        mass_def_500c = MassDefinition(500, "critical")
        c_old = halo_model.concentration.c_delta(halo_model, m, z)
        m500c = convert_m_delta(halo_model.cosmology, m, z, mass_def_old, mass_def_500c, c_old=c_old)

        r_500c = mass_def_500c.r_delta(halo_model.cosmology, m500c, z)  # (Nm, Nz), physical Mpc
        x_500c = r[:, None, None] / ((1.0 + z[None, None, :]) * r_500c[None, :, :])  # (Nr, Nm, Nz)

        H = jnp.atleast_1d(halo_model.cosmology.hubble_parameter(z))[None, None, :]  # (1, 1, Nz)
        E_z = H / H0
        m500c_tilde = (m500c * h / B)[None]  # (1, Nm, Nz)

        # ---- Arnaud P_500c (same as GNFWPressureProfile) ----
        P_500c_arnaud = (
            1.65 * (h / 0.7) ** 2 * E_z ** (8.0 / 3.0)
            * (m500c_tilde / (0.7 * 3e14)) ** (2.0 / 3.0 + 0.12)
            * (0.7 / h) ** 1.5
        )  # (1, Nm, Nz)

        # ---- Compton-y0 amplitudes (matches tszpower compute_y0 / compute_y0_parametric) ----
        sigma_T_cm2 = 6.6524587e-25                     # Thomson cross section in cm^2
        m_e_c2_eV = 510998.95                           # m_e c^2 in eV
        shape_integral = 0.470502095                    # GNFW shape integral, hardcoded as in tszpower
        mpc_to_cm = Const._Mpc_over_m_ * 100.0          # cm per Mpc (physical)

        r_500c_cm = (r_500c * mpc_to_cm)[None, :, :]    # (1, Nm, Nz), physical cm

        y0_orig = (
            2.0 * (sigma_T_cm2 / m_e_c2_eV)
            * P0 * P_500c_arnaud * r_500c_cm
            * shape_integral
        )                                                # (1, Nm, Nz)

        y0_param = (
            (10.0 ** A_SZ)
            * (m500c_tilde / (0.7 * 3e14)) ** alpha_SZ
            * E_z ** 2
            * (h / 0.7) ** (-0.5)
        )                                                # (1, Nm, Nz)

        ratio = y0_param / y0_orig                       # (1, Nm, Nz)

        scaled_x = c500 * x_500c
        Pe_arnaud = P_500c_arnaud * P0 * scaled_x ** (-gamma) * (1 + scaled_x ** alpha) ** ((gamma - beta) / alpha)
        Pe = Pe_arnaud * ratio                           # (Nr, Nm, Nz)

        return Pe


jax.tree_util.register_pytree_node(
    ParametricGNFWPressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: ParametricGNFWPressureProfile._tree_unflatten(aux_data, children)
)


class TruncatedParametricGNFWPressureProfile(ParametricGNFWPressureProfile):
    r"""
    :class:`ParametricGNFWPressureProfile` with a per-halo angular truncation
    matching the JXPaint painter's disc cutoff.

    The painter assigns Compton-:math:`y` to sky pixels within an angular
    radius :math:`\theta_{\max}(M, z)` of each halo centre,

    .. math::

        \theta_{\max}(M, z) = \min\!\Big(
        \max\big(2\,\mathrm{FWHM},\; c_{\rm mult}\,\theta_{500}\big),\;
        \theta_{\rm cap}\Big),
        \qquad
        \theta_{500} = \arctan\!\frac{r_{500c} / B^{1/3}}{d_A(z)},
        \tag{1}

    where :math:`r_{500c}` is the physical :math:`500c` radius and
    :math:`B` is the hydrostatic mass bias (geometry matches
    ``jxpaint.painting.geometry.theta_max_array``).  The truncation is a
    cylinder (line-of-sight disc) cutoff of the projected profile, so it is
    folded into the harmonic-space profile as a multiplicative ratio

    .. math::

        u_\ell^{\rm trunc}(k, M, z) = u_\ell^{\rm full}(k, M, z)\,
        T(k, M, z),
        \qquad
        T = \frac{\mathcal{H}_0[y_{2D}\,\Theta(\theta \le \theta_{\max})]}
                 {\mathcal{H}_0[y_{2D}]},
        \tag{2}

    where :math:`y_{2D}` is the line-of-sight projected profile,
    :math:`\mathcal{H}_0` a Hankel transform of order zero, and
    :math:`u_\ell^{\rm full}` is the parent's exact profile so the absolute
    normalization is unchanged.  :math:`T` is precomputed on a coarse
    :math:`(M, z, q)` grid (:math:`q = k\,r_{500c}^{\rm com}` dimensionless)
    by :meth:`precompute` and trilinearly interpolated inside :meth:`u_k`.

    Notes
    -----
    Because the full profile is the parent's exact :math:`u_k`, any halo-model
    integral that calls :meth:`u_k` (e.g. ``cl_1h_masked``,
    ``trispectrum_1h_masked``) is truncated consistently.  The :math:`q`-axis
    uses ``MassDefinition(500, "critical").r_delta`` on the input mass, so the
    truncation is exact when the halo model's mass definition is :math:`500c`
    (the JXPaint validation setup).

    Parameters
    ----------
    fwhm_arcmin : float
        Beam FWHM in arcmin (sets the :math:`2\,\mathrm{FWHM}` floor).
    cap_deg : float
        Angular cap :math:`\theta_{\rm cap}` in degrees.
    mult : float
        Multiple :math:`c_{\rm mult}` of :math:`\theta_{500}` (painter uses 4).
    n_mz : int
        Number of grid points per axis for the coarse :math:`(M, z)` cache.
    nperp, ns, nr : int
        Sampling of the projected-radius, line-of-sight, and 3D-radius grids.
    """

    def __init__(self, x=None, A_SZ=-4.97, alpha_SZ=0.7867,
                 P0=8.130, c500=1.156, alpha=1.0620, beta=5.4807, gamma=0.3292,
                 B=1.4, fwhm_arcmin=10.0, cap_deg=5.0, mult=4.0,
                 n_mz=40, n_theta=3000, nq=512, **_legacy):
        super().__init__(x=x, A_SZ=A_SZ, alpha_SZ=alpha_SZ, P0=P0, c500=c500,
                         alpha=alpha, beta=beta, gamma=gamma, B=B)
        self.fwhm_arcmin = float(fwhm_arcmin)
        self.cap_deg = float(cap_deg)
        self.mult = float(mult)
        self.n_mz = int(n_mz)
        self.n_theta = int(n_theta)
        self.nq = int(nq)
        self._cache = None  # filled by precompute()

    def precompute(self, halo_model, m_min, m_max, z_min, z_max):
        r"""
        Tabulate the truncation ratio :math:`T(k, M, z)` on a coarse grid.

        The painter smears each halo with a Gaussian beam, truncates the
        *beamed* projected profile at the angular disc radius
        :math:`\theta_{\max}`, and the power-spectrum pipeline then deconvolves
        the beam.  Because beam-convolution and a sharp disc cut do not commute,
        the truncation ratio is computed to match that exact sequence:

        .. math::

            T(\ell, M, z) = \frac{\mathcal{H}_0\!\big[(y_{2D} \ast b)\,
            \Theta(\theta \le \theta_{\max})\big] / b_\ell}
            {\mathcal{H}_0[y_{2D}]},

        where :math:`y_{2D}(\theta) = Y(\theta / \theta_{500})` is the projected
        GNFW shape on the hydrostatically de-biased angular scale
        :math:`\theta_{500} = (r_{500c}/B^{1/3})/d_A` (matching the painter and
        ``PressureProfile.u_k``), :math:`b` is the Gaussian beam of FWHM
        ``fwhm_arcmin``, and :math:`b_\ell` its harmonic transform.  Setting
        ``fwhm_arcmin = 0`` recovers the sharp-truncation ratio.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing cosmology and radii. Its mass definition is
            treated as :math:`500c` for the :math:`q`-axis scaling.
        m_min, m_max : float
            Mass range (same units as the halo model mass) for the cache.
        z_min, z_max : float
            Redshift range for the cache.

        Returns
        -------
        TruncatedParametricGNFWPressureProfile
            ``self``, with the interpolation cache populated.

        Notes
        -----
        All Hankel/beam work lives here (run once).  ``u_k`` only interpolates
        the cached :math:`T` and multiplies the parent profile, so the per-call
        power-spectrum cost is unchanged.  :math:`T` is a pure shape ratio and
        is independent of ``A_SZ``/``alpha_SZ``, so it need not be recomputed
        when only the tSZ amplitude parameters vary.
        """
        cosmo = halo_model.cosmology
        Nm = Nz = self.n_mz
        mg = np.geomspace(m_min, m_max, Nm)
        zg = np.geomspace(z_min, z_max, Nz)

        mdef500 = MassDefinition(500, "critical")
        R500 = np.asarray(mdef500.r_delta(cosmo, jnp.asarray(mg), jnp.asarray(zg)))  # (Nm,Nz)
        d_A = np.asarray(cosmo.angular_diameter_distance(jnp.asarray(zg)))[None, :]  # (1,Nz)
        th500b = (R500 / self.B ** (1.0 / 3.0)) / d_A                  # biased angular R500
        th500u = R500 / d_A                                            # unbiased (q<->ell map)

        # Painter theta_max: min(max(2FWHM, mult*theta500b), cap).
        two_fwhm = 2.0 * np.deg2rad(self.fwhm_arcmin / 60.0)
        theta_max = np.minimum(np.maximum(two_fwhm, self.mult * th500b),
                               np.deg2rad(self.cap_deg))               # (Nm,Nz)

        # Universal projected GNFW shape Y(u) = 2 int_0^inf gnfw(sqrt(u^2+w^2)) dw
        # (tanh-sinh), matching the biased-scale profile that PressureProfile.u_k
        # encodes; amplitude is irrelevant since T is a ratio.
        c500, al, be, ga = self.c500, self.alpha, self.beta, self.gamma
        ts_t = np.linspace(-4.0, 4.0, 600)
        ts_w = np.exp((np.pi / 2.0) * np.sinh(ts_t))
        ts_q = (ts_t[1] - ts_t[0]) * ts_w * (np.pi / 2.0) * np.cosh(ts_t)
        ug = np.logspace(-7, 5, 30000)
        sc = c500 * np.sqrt(ug[:, None] ** 2 + ts_w[None, :] ** 2)
        Yg = 2.0 * np.sum(sc ** (-ga) * (1.0 + sc ** al) ** ((ga - be) / al) * ts_q[None, :], axis=1)

        def _Y(x):
            return np.interp(np.log(np.clip(x, ug[0], ug[-1])), np.log(ug), Yg,
                             left=Yg[0], right=0.0)

        # Fixed angular grid (out past the cap plus beam wings) and projected
        # profile per (M,z): y2d(theta) = Y(theta / theta500b).
        theta = np.logspace(-6, np.log10(np.deg2rad(2.0 * self.cap_deg + 4.0)), self.n_theta)
        y2d = _Y(theta[:, None, None] / th500b[None, :, :])           # (Nt,Nm,Nz)
        Y2 = y2d.reshape(self.n_theta, -1)                            # (Nt, Nm*Nz)

        # Beam the projected profile (flat-sky Gaussian, exponentially scaled I0
        # for stability), truncate the beamed disc, then deconvolve b_ell.
        sig = np.deg2rad(self.fwhm_arcmin / 60.0) / 2.3548200450309493
        if sig > 0.0:
            s2 = sig * sig
            ti = theta[:, None]
            tj = theta[None, :]
            dt = np.gradient(theta)[None, :]
            W = ive(0, ti * tj / s2) * np.exp(-(ti - tj) ** 2 / (2.0 * s2)) * (tj / s2) * dt
            W = W / np.clip(W.sum(axis=1, keepdims=True), 1e-300, None)
            Y2b = W @ Y2
        else:
            Y2b = Y2
        disc = (theta[:, None] <= theta_max.reshape(-1)[None, :]).astype(np.float64)
        Y2bt = Y2b * disc

        han = mcfit.Hankel(theta, nu=0, lowring=True)
        ell_native, F_full = han(Y2.T)
        _, F_bt = han(Y2bt.T)
        F_full = F_full.T                                            # (Nell, Nm*Nz)
        F_bt = F_bt.T
        bl = np.exp(-0.5 * ell_native ** 2 * sig * sig) if sig > 0.0 else np.ones_like(ell_native)
        T_ell = (F_bt / (bl[:, None] * np.where(np.abs(F_full) > 0, F_full, 1.0)))
        T_ell = T_ell.reshape(len(ell_native), Nm, Nz)

        # Recast T(ell) onto the q = k*R500_c axis used by u_k: ell = q / theta500u.
        q_native = np.geomspace(1e-4, 1e2, self.nq)
        log_ell = np.log(ell_native)
        T_cache = np.empty((Nm, Nz, self.nq))
        for im in range(Nm):
            for iz in range(Nz):
                ellq = q_native / th500u[im, iz]
                T_cache[im, iz] = np.interp(
                    np.log(np.clip(ellq, ell_native[0], ell_native[-1])),
                    log_ell, T_ell[:, im, iz], left=T_ell[0, im, iz], right=1.0)

        self._cache = {
            "logm_grid": jnp.asarray(np.log(mg)),
            "z_grid": jnp.asarray(zg),
            "q_native": jnp.asarray(q_native),
            "T_cache": jnp.asarray(T_cache),
        }
        return self

    def _interp_T(self, logm, z, q):
        """Trilinear interpolation of the cached T over (logm, z, q)."""
        lg = self._cache["logm_grid"]
        zg = self._cache["z_grid"]
        qg = self._cache["q_native"]
        Tc = self._cache["T_cache"]                                  # (Nm_c,Nz_c,Nq)

        def idx1d(xp, x):
            i = jnp.searchsorted(xp, x, side="right") - 1
            return jnp.clip(i, 0, xp.shape[0] - 2)

        il = idx1d(lg, logm)                                         # (Nm,)
        iz = idx1d(zg, z)                                            # (Nz,)
        iq = idx1d(qg, q)                                            # (Nk,Nm,Nz)
        wl = (logm - lg[il]) / (lg[il + 1] - lg[il])                # (Nm,)
        wz = (z - zg[iz]) / (zg[iz + 1] - zg[iz])                   # (Nz,)
        wq = (q - qg[iq]) / (qg[iq + 1] - qg[iq])                   # (Nk,Nm,Nz)

        il_b = il[None, :, None]
        iz_b = iz[None, None, :]
        iq_b = iq
        wl_b = wl[None, :, None]
        wz_b = wz[None, None, :]
        c000 = Tc[il_b, iz_b, iq_b];         c001 = Tc[il_b, iz_b, iq_b + 1]
        c010 = Tc[il_b, iz_b + 1, iq_b];     c011 = Tc[il_b, iz_b + 1, iq_b + 1]
        c100 = Tc[il_b + 1, iz_b, iq_b];     c101 = Tc[il_b + 1, iz_b, iq_b + 1]
        c110 = Tc[il_b + 1, iz_b + 1, iq_b]; c111 = Tc[il_b + 1, iz_b + 1, iq_b + 1]
        T = (c000 * (1 - wl_b) * (1 - wz_b) * (1 - wq) + c001 * (1 - wl_b) * (1 - wz_b) * wq +
             c010 * (1 - wl_b) * wz_b * (1 - wq)       + c011 * (1 - wl_b) * wz_b * wq +
             c100 * wl_b * (1 - wz_b) * (1 - wq)       + c101 * wl_b * (1 - wz_b) * wq +
             c110 * wl_b * wz_b * (1 - wq)             + c111 * wl_b * wz_b * wq)
        T = jnp.where(q > qg[-1], 1.0, T)                           # high-q -> no truncation
        T_lo = Tc[il_b, iz_b, 0]
        T = jnp.where(q < qg[0], T_lo, T)                           # low-q -> ell->0 limit
        return T

    @jax.jit
    def u_k(self, halo_model, k, m, z):
        """
        Parent :meth:`ParametricGNFWPressureProfile.u_k` with the disc
        truncation folded in.

        Returns
        -------
        jnp.ndarray
            Truncated Fourier-space profile with shape :math:`(N_k, N_m, N_z)`.
            Reduces to the parent profile when :meth:`precompute` has not run.
        """
        u_full = super().u_k(halo_model, k, m, z)                   # (Nk,Nm,Nz) exact
        if self._cache is None:
            return u_full
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        mdef500 = MassDefinition(500, "critical")
        R500_c = mdef500.r_delta(halo_model.cosmology, m, z) * (1.0 + z)[None, :]  # (Nm,Nz)
        q = k[:, None, None] * R500_c[None, :, :]                   # (Nk,Nm,Nz)
        T = self._interp_T(jnp.log(m), z, q)
        return u_full * T

    def _tree_flatten(self):
        c = self._cache
        if c is None:                              # before precompute()
            z1 = jnp.zeros((1,))
            logm_g = z_g = q_g = Tc = z1
        else:
            logm_g, z_g, q_g, Tc = (
                c["logm_grid"], c["z_grid"], c["q_native"], c["T_cache"])
        leaves = (self.A_SZ, self.alpha_SZ, self.P0, self.c500, self.alpha,
                  self.beta, self.gamma, self.B,
                  logm_g, z_g, q_g, Tc)
        aux_data = (tuple(self._x.tolist()), self._hankel, self.fwhm_arcmin,
                    self.cap_deg, self.mult, self.n_mz, self.n_theta, self.nq,
                    c is not None)
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        (x_tuple, hankel, fwhm, cap, mult, nmz, n_theta, nq, has_cache) = aux_data
        (A_SZ, alpha_SZ, P0, c500, alpha, beta, gamma, B,
         logm_grid, z_grid, q_native, T_cache) = leaves
        obj = cls.__new__(cls)
        (obj.A_SZ, obj.alpha_SZ, obj.P0, obj.c500, obj.alpha,
         obj.beta, obj.gamma, obj.B) = (A_SZ, alpha_SZ, P0, c500, alpha, beta, gamma, B)
        obj._x = np.array(x_tuple)
        obj._hankel = hankel
        obj.fwhm_arcmin, obj.cap_deg, obj.mult = fwhm, cap, mult
        obj.n_mz, obj.n_theta, obj.nq = nmz, n_theta, nq
        obj._cache = {
            "logm_grid": logm_grid, "z_grid": z_grid, "q_native": q_native,
            "T_cache": T_cache,
        } if has_cache else None
        return obj


jax.tree_util.register_pytree_node(
    TruncatedParametricGNFWPressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: TruncatedParametricGNFWPressureProfile._tree_unflatten(aux_data, children)
)


class TruncatedGNFWPressureProfile(GNFWPressureProfile):
    """Ordinary Arnaud GNFW profile with the painter-matched disc truncation.

    Its pressure normalization and JAX state are purely those of
    :class:`GNFWPressureProfile`; only the precomputed beam-and-disc transfer
    function is shared with the truncated parametric implementation.
    """

    def __init__(self, x=None, P0=8.130, c500=1.156, alpha=1.0620,
                 beta=5.4807, gamma=0.3292, B=1.4, fwhm_arcmin=10.0,
                 cap_deg=5.0, mult=4.0, n_mz=40, n_theta=3000, nq=512,
                 **legacy):
        super().__init__(x=x, P0=P0, c500=c500, alpha=alpha, beta=beta,
                         gamma=gamma, B=B)
        self.fwhm_arcmin = float(fwhm_arcmin)
        self.cap_deg = float(cap_deg)
        self.mult = float(mult)
        self.n_mz = int(n_mz)
        self.n_theta = int(n_theta)
        self.nq = int(nq)
        self._cache = None

    precompute = TruncatedParametricGNFWPressureProfile.precompute
    _interp_T = TruncatedParametricGNFWPressureProfile._interp_T

    @jax.jit
    def u_k(self, halo_model, k, m, z):
        u_full = GNFWPressureProfile.u_k(self, halo_model, k, m, z)
        if self._cache is None:
            return u_full
        k, m, z = jnp.atleast_1d(k), jnp.atleast_1d(m), jnp.atleast_1d(z)
        mdef500 = MassDefinition(500, "critical")
        R500_c = mdef500.r_delta(
            halo_model.cosmology, m, z
        ) * (1.0 + z)[None, :]
        q = k[:, None, None] * R500_c[None, :, :]
        return u_full * self._interp_T(jnp.log(m), z, q)

    def _tree_flatten(self):
        c = self._cache
        if c is None:
            z1 = jnp.zeros((1,))
            logm_g = z_g = q_g = Tc = z1
        else:
            logm_g, z_g, q_g, Tc = (
                c["logm_grid"], c["z_grid"], c["q_native"], c["T_cache"])
        leaves = (self.P0, self.c500, self.alpha, self.beta, self.gamma, self.B,
                  logm_g, z_g, q_g, Tc)
        aux_data = (tuple(self._x.tolist()), self._hankel, self.fwhm_arcmin,
                    self.cap_deg, self.mult, self.n_mz, self.n_theta, self.nq,
                    c is not None)
        return leaves, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        (x_tuple, hankel, fwhm, cap, mult, nmz, n_theta, nq,
         has_cache) = aux_data
        (P0, c500, alpha, beta, gamma, B,
         logm_grid, z_grid, q_native, T_cache) = leaves
        obj = cls.__new__(cls)
        (obj.P0, obj.c500, obj.alpha, obj.beta, obj.gamma, obj.B) = (
            P0, c500, alpha, beta, gamma, B)
        obj._x = np.array(x_tuple)
        obj._hankel = hankel
        obj.fwhm_arcmin, obj.cap_deg, obj.mult = fwhm, cap, mult
        obj.n_mz, obj.n_theta, obj.nq = nmz, n_theta, nq
        obj._cache = {
            "logm_grid": logm_grid, "z_grid": z_grid,
            "q_native": q_native, "T_cache": T_cache,
        } if has_cache else None
        return obj


jax.tree_util.register_pytree_node(
    TruncatedGNFWPressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: TruncatedGNFWPressureProfile._tree_unflatten(
        aux_data, children
    ),
)


class B12PressureProfile(PressureProfile):
    """
    Electron pressure profile from `Battaglia et al. (2012) <https://ui.adsabs.harvard.edu/abs/2012ApJ...758...74B/abstract>`_.

    The profile is evaluated as a function of the comoving radius
    :math:`r`, but its normalization and shape are defined using the native
    :math:`200c` calibration mass and radius:

    .. math::

        P_e(r, M, z)
        = P_{200c} \\, P_0
        \\left(\\frac{x_{200c}}{x_c}\\right)^\\gamma
        \\left[1 + \\left(\\frac{x_{200c}}{x_c}\\right)^\\alpha\\right]^{-\\beta}
        \\tag{1}

    where :math:`x_{200c} = r / r_{200c}` and :math:`r_{200c}` has the same
    units as :math:`r`.
    In this implementation, :math:`\\alpha = 1` and :math:`\\gamma = -0.3`,
    and the remaining profile parameters follow the Battaglia scaling

    .. math::

        X(M_{200c}, z) = A_X
        \\left(\\frac{M_{200c} / h}{10^{14} M_\\odot}\\right)^{\\alpha_m^X}
        (1 + z)^{\\alpha_z^X}
        \\tag{2}

    where :math:`X \\in \\{P_0, x_c, \\beta\\}`. In the implementation, the
    input halo mass is first converted from the halo model's mass definition to
    :math:`M_{200c}`.

    The projected Fourier-space pressure profile is evaluated as

    .. math::

        u_\\ell(\\ell, M, z) =
        \\frac{4 \\pi (1+z) r_\\Delta}{\\ell_\\Delta^2}
        \\int dx \\, x^2 \\, P_e(x, M, z)
        \\, \\frac{\\sin\\!\\left[(\\ell / \\ell_\\Delta) x\\right]}
        {(\\ell / \\ell_\\Delta) x}
        \\tag{3}

    where :math:`\\ell_\\Delta(M, z) = d_A(z) / r_\\Delta(M, z)` and
    :math:`\\chi(z) = (1+z) d_A(z)`.

    Attributes
    ----------
    x : jnp.ndarray
        Dimensionless radial grid :math:`x = r / r_\\Delta` used to tabulate the profile and define the Hankel transform, with :math:`r_\\Delta` expressed in the same units as :math:`r`.
    A_P0 : float
        Amplitude :math:`A_{P_0}` of the pressure normalization scaling.
    A_xc : float
        Amplitude :math:`A_{x_c}` of the core-radius scaling.
    A_beta : float
        Amplitude :math:`A_\\beta` of the outer-slope scaling.
    alpha_m_P0 : float
        Mass-scaling exponent :math:`\\alpha_m^{P_0}`.
    alpha_m_xc : float
        Mass-scaling exponent :math:`\\alpha_m^{x_c}`.
    alpha_m_beta : float
        Mass-scaling exponent :math:`\\alpha_m^\\beta`.
    alpha_z_P0 : float
        Redshift-scaling exponent :math:`\\alpha_z^{P_0}`.
    alpha_z_xc : float
        Redshift-scaling exponent :math:`\\alpha_z^{x_c}`.
    alpha_z_beta : float
        Redshift-scaling exponent :math:`\\alpha_z^\\beta`.
    """
    def __init__(self, x=None, 
                 A_P0=18.1, A_xc=0.497, A_beta=4.35,
                 alpha_m_P0=0.154, alpha_m_xc=-0.00865, alpha_m_beta=0.0393,
                 alpha_z_P0=-0.758, alpha_z_xc=0.731, alpha_z_beta=0.415):
        
        # Physics Parameters (The Leaves)
        self.A_P0, self.A_xc, self.A_beta = A_P0, A_xc, A_beta
        self.alpha_m_P0, self.alpha_m_xc, self.alpha_m_beta = alpha_m_P0, alpha_m_xc, alpha_m_beta
        self.alpha_z_P0, self.alpha_z_xc, self.alpha_z_beta = alpha_z_P0, alpha_z_xc, alpha_z_beta

        # Grid initialization: store as numpy for hashable PyTree aux_data
        self.x = x if x is not None else np.logspace(-4, 1, 256)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.asarray(value)
        self._hankel = HankelTransform(jnp.asarray(self._x), nu=0.5)

    def _tree_flatten(self):
        leaves = (
            self.A_P0, self.A_xc, self.A_beta,
            self.alpha_m_P0, self.alpha_m_xc, self.alpha_m_beta,
            self.alpha_z_P0, self.alpha_z_xc, self.alpha_z_beta,
        )
        aux_data = (tuple(self._x.tolist()), self._hankel)
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        x_tuple, hankel = aux_data
        obj = cls.__new__(cls)

        (obj.A_P0, obj.A_xc, obj.A_beta,
         obj.alpha_m_P0, obj.alpha_m_xc, obj.alpha_m_beta,
         obj.alpha_z_P0, obj.alpha_z_xc, obj.alpha_z_beta) = leaves

        obj._x = np.array(x_tuple)
        obj._hankel = hankel
        return obj

    def update(self, A_P0=None, A_xc=None, A_beta=None,
               alpha_m_P0=None, alpha_m_xc=None, alpha_m_beta=None,
               alpha_z_P0=None, alpha_z_xc=None, alpha_z_beta=None):
        """
        Return a new profile instance with updated B12 parameters.
    
        Parameters
        ----------
        A_P0, A_xc, A_beta, alpha_m_P0, alpha_m_xc, alpha_m_beta, alpha_z_P0, alpha_z_xc, alpha_z_beta : float, optional
            Replacement values for the corresponding class attributes. Any argument left as ``None`` keeps its current value.
    
        Returns
        -------
        B12PressureProfile
            New profile instance with updated parameters.
        """
        leaves, treedef = self._tree_flatten()
        
        new_leaves = (
            A_P0 if A_P0 is not None else self.A_P0,
            A_xc if A_xc is not None else self.A_xc,
            A_beta if A_beta is not None else self.A_beta,
            alpha_m_P0 if alpha_m_P0 is not None else self.alpha_m_P0,
            alpha_m_xc if alpha_m_xc is not None else self.alpha_m_xc,
            alpha_m_beta if alpha_m_beta is not None else self.alpha_m_beta,
            alpha_z_P0 if alpha_z_P0 is not None else self.alpha_z_P0,
            alpha_z_xc if alpha_z_xc is not None else self.alpha_z_xc,
            alpha_z_beta if alpha_z_beta is not None else self.alpha_z_beta,
        )
        
        return self._tree_unflatten(treedef, new_leaves)

    @jax.jit
    def u_r(self, halo_model, r, m, z):
        """
        Compute the electron-pressure profile.

        Parameters
        ----------
        halo_model : HaloModel
            Halo model providing the cosmology, mass-definition conversion, and halo
            radius.
        r : float or jnp.ndarray
            Comoving radius or radii in :math:`\\mathrm{Mpc}`.
        m : float or jnp.ndarray
            Halo mass or masses in physical :math:`M_\\odot`.
        z : float or jnp.ndarray
            Redshift(s).

        Returns
        -------
        jnp.ndarray
            Electron pressure profile with shape :math:`(N_r, N_m, N_z)`.
        """
        cparams = halo_model.cosmology._cosmo_params()
        h = cparams["h"]
        alpha, gamma = 1.0, -0.3
        r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m), jnp.atleast_1d(z)
    
        # Convert input mass to M200c for normalization
        mass_def_old = halo_model.mass_definition
        mass_def_200c = MassDefinition(200, "critical")
        c_old = halo_model.concentration.c_delta(halo_model, m, z)
        m200c = convert_m_delta(halo_model.cosmology, m, z, mass_def_old, mass_def_200c, c_old=c_old)
    
        r_200c = mass_def_200c.r_delta(halo_model.cosmology, m200c, z)  # (Nm, Nz)
    
        # Convert the comoving radius to the calibrated physical 200c coordinate.
        x_200c = r[:, None, None] / ((1.0 + z[None, None, :]) * r_200c[None, :, :])  # (Nr, Nm, Nz)
        m200c_b = m200c[None]
        z_b = z[None, None, :]
        mass_ratio = m200c_b / 1e14
    
        # Compute shape parameters using M200c
        P0 = self.A_P0 * mass_ratio**self.alpha_m_P0 * (1 + z_b)**self.alpha_z_P0
        xc = self.A_xc * mass_ratio**self.alpha_m_xc * (1 + z_b)**self.alpha_z_xc
        beta = self.A_beta * mass_ratio**self.alpha_m_beta * (1 + z_b)**self.alpha_z_beta
    
        # Normalized GNFW shape
        scaled_x = x_200c / xc
        p_x = (scaled_x)**gamma * (1 + scaled_x**alpha)**(-beta)
    
        # Thermal Pressure Normalization (P200c)
        H = jnp.atleast_1d(halo_model.cosmology.hubble_parameter(z))
        f_b = cparams["Omega_b"] / cparams["Omega0_m"]
        r_200c = r_200c * h
        # Use M200c and r_200c for normalization
        P_200c = ((m200c_b / r_200c[None, :, :]) * f_b * 2.61051e-18 * (H[None, None, :])**2)
    
        return P_200c * P0 * p_x

jax.tree_util.register_pytree_node(
    B12PressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: B12PressureProfile._tree_unflatten(aux_data, children)
)
