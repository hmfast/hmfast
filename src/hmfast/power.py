import functools

import jax
import jax.numpy as jnp
import mcfit
import numpy as np

from hmfast.halos.profiles.profiles_2pt import _fourier_2pt


# -------------------------
# Perturbation theory helpers
# -------------------------
def _F2(k1, k2, mu):
    """Leading-order SPT kernel F2 (EdS approximation)."""
    k1_over_k2 = jnp.where(k2 == 0.0, 0.0, k1 / k2)
    k2_over_k1 = jnp.where(k1 == 0.0, 0.0, k2 / k1)
    return 5.0 / 7.0 + 0.5 * (k1_over_k2 + k2_over_k1) * mu + 2.0 / 7.0 * mu ** 2

@jax.jit
def _mu(k1, k2, k3):
    """Cosine between k1 and k2 given opposite side k3."""
    den = 2.0 * k1 * k2
    return jnp.where(
        den == 0.0,
        0.0,
        (k3 ** 2 - k1 ** 2 - k2 ** 2) / den
    )

@jax.jit
def _ksum(k1, k2, mu):
    """Magnitude of the vector sum of k1, k2 given the cosine of the angle between them."""
    return jnp.sqrt(k1 ** 2 + k2 ** 2 + 2.0 * k1 * k2 * mu)


# Trispectrum tree-level helpers (4-halo term)
#
# The 4-halo trispectrum term needs the tree-level kernel angle-averaged over
# the relative orientation of the k_u and k_v leg pairs -- a genuinely free
# angle in the parallelogram configuration, unlike the bispectrum's triangle
# angle which is fixed by k1,k2,k3. This is new numerical machinery not
# needed elsewhere in hmfast: a fixed-order Gauss-Legendre quadrature over
# theta in [0, pi], evaluating the tree-level trispectrum kernel of
# Takada & Hu (2013), Eq. 30 (arXiv:1302.6994).
_TRISPEC_N_THETA = 96
_trispec_gl_x, _trispec_gl_w = np.polynomial.legendre.leggauss(_TRISPEC_N_THETA)
_trispec_gl_x = jnp.asarray(_trispec_gl_x)
_trispec_gl_w = jnp.asarray(_trispec_gl_w)
_TRISPEC_COS_THETA = jnp.cos(0.5 * jnp.pi * (_trispec_gl_x + 1.0))
_TRISPEC_THETA_WEIGHT = _trispec_gl_w * (jnp.pi / 2.0) / jnp.pi


@jax.jit
def _X3(k, kp):
    """
    Closed-form, angle-averaged tree-level F3-type kernel entering the
    "1113" diagram of the 4-halo trispectrum, following Eq. 30 of
    Takada & Hu (2013). Depends only on the ratio r = kp/k.
    """
    r = (kp / k)[:, None]
    cth = _TRISPEC_COS_THETA[None, :]
    wgt = _TRISPEC_THETA_WEIGHT[None, :]

    kr = _ksum(k[:, None], kp[:, None], cth)
    intd = (
        (5.0 * r + (7.0 - 2.0 * r ** 2) * cth) / (1.0 + r ** 2 + 2.0 * r * cth)
        * (3.0 / 7.0 * r + 0.5 * (1.0 + r ** 2) * cth + 4.0 / 7.0 * r * cth ** 2)
    )
    intd = jnp.where(kr == 0.0, 0.0, intd)

    isotropized = jnp.sum(intd * wgt, axis=1)
    return -7.0 / 4.0 * (1.0 + jnp.squeeze(r, axis=1) ** 2) + isotropized



# -------------------------
# Halo model mass-integral building blocks
# -------------------------
#
# Shared across Bk and Tk (not tied to either class's state), so kept at
# module level rather than duplicated as a private method on each.

def _pair_integral(halo_model, p1, p2, k1, k2, z):
    """
    ∫ dn/dlnM * b1(M) * p1.fourier(k1,M,z) * p2.fourier(k2,M,z) dlnM

    Pair integral with first-order halo bias included.
    Used as a building block for ``Bk.bk_2h`` and
    the halo-model trispectrum's 2h/3h terms.

    Future generalisation: replace ``u1 * u2`` with a ``_fourier_2pt``
    variant that handles different k values when specialised 2-point kernels
    (HOD, CIB) are needed.
    """
    hm = halo_model
    m, z_arr = hm.m_grid, jnp.atleast_1d(z)
    logm = jnp.log(m)
    dm = jnp.diff(logm)
    w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

    dndlnm = jnp.reshape(
        hm.halo_mass_function.dndlnm(hm.cosmology, m, z_arr, hm.mass_def, hm.convert_masses),
        (len(m), len(z_arr)),
    )
    bias_w = jnp.reshape(
        hm.halo_bias.bias(hm.cosmology, m, z_arr, hm.mass_def, hm.convert_masses, order=1),
        (len(m), len(z_arr)),
    )
    total_weights = dndlnm * bias_w * w[:, None]  # (Nm, Nz)

    k1s, k2s = jnp.atleast_1d(k1), jnp.atleast_1d(k2)
    u1 = jnp.reshape(p1.fourier(hm, k1s, m, z_arr), (len(k1s), len(m), len(z_arr)))
    u2 = jnp.reshape(p2.fourier(hm, k2s, m, z_arr), (len(k2s), len(m), len(z_arr)))

    integral = jnp.sum(u1 * u2 * total_weights[None, :, :], axis=1)  # (Nk, Nz)

    n_min, b1_min, _ = hm._counter_terms(z_arr)
    correction = n_min[None, :] * b1_min[None, :] * u1[:, 0, :] * u2[:, 0, :]

    return jnp.squeeze(integral + hm.hm_consistency * correction)


def _triple_integral(halo_model, p_single, p_pair1, p_pair2, k_single, k_pair, z):
    """
    ∫ dn/dlnM * b1(M) * p_single(k_single,M) * p_pair1(k_pair,M) * p_pair2(k_pair,M) dlnM

    Triple integral with first-order halo bias, generalising ``_pair_integral``
    to a "1x2" moment: one profile alone at ``k_single``, the other two both
    at ``k_pair``. Needed by the trispectrum's 2-halo "13" term.
    """
    hm = halo_model
    m, z_arr = hm.m_grid, jnp.atleast_1d(z)
    logm = jnp.log(m)
    dm = jnp.diff(logm)
    w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

    dndlnm = jnp.reshape(
        hm.halo_mass_function.dndlnm(hm.cosmology, m, z_arr, hm.mass_def, hm.convert_masses),
        (len(m), len(z_arr)),
    )
    bias_w = jnp.reshape(
        hm.halo_bias.bias(hm.cosmology, m, z_arr, hm.mass_def, hm.convert_masses, order=1),
        (len(m), len(z_arr)),
    )
    total_weights = dndlnm * bias_w * w[:, None]  # (Nm, Nz)

    k_s, k_p = jnp.atleast_1d(k_single), jnp.atleast_1d(k_pair)
    u_s = jnp.reshape(p_single.fourier(hm, k_s, m, z_arr), (len(k_s), len(m), len(z_arr)))
    u_p1 = jnp.reshape(p_pair1.fourier(hm, k_p, m, z_arr), (len(k_p), len(m), len(z_arr)))
    u_p2 = jnp.reshape(p_pair2.fourier(hm, k_p, m, z_arr), (len(k_p), len(m), len(z_arr)))

    integral = jnp.sum(u_s * u_p1 * u_p2 * total_weights[None, :, :], axis=1)  # (Nk, Nz)

    n_min, b1_min, _ = hm._counter_terms(z_arr)
    correction = n_min[None, :] * b1_min[None, :] * u_s[:, 0, :] * u_p1[:, 0, :] * u_p2[:, 0, :]

    return jnp.squeeze(integral + hm.hm_consistency * correction)


def _kr_pkr(hm, k, kp, z_arr):
    """
    Shared per-theta ``kr = |k + kp|`` and :math:`P_{\\mathrm{lin}}(k_r)`
    arrays entering every angle-averaged trispectrum kernel
    (``_Pbar_kernel``, ``_P3_kernel``, ``_P4_kernel``). Computing this once
    and reusing it avoids repeating the (relatively expensive) emulator
    evaluation of :math:`P_{\\mathrm{lin}}` across the 2h, 3h and 4h terms.

    ``k``, ``kp`` : (N,) arrays. Returns ``(k_b, kp_b, kr, pkr)`` with
    ``k_b``, ``kp_b`` of shape (N,1) and ``kr``, ``pkr`` of shape
    (N, N_theta).
    """
    k_b, kp_b = k[:, None], kp[:, None]
    cth = _TRISPEC_COS_THETA[None, :]
    kr = _ksum(k_b, kp_b, cth)
    pkr = jnp.reshape(hm.cosmology.pk(kr.flatten(), z_arr, linear=True), kr.shape)
    return k_b, kp_b, kr, pkr


# -------------------------
# Halo model power spectrum
# -------------------------

class Pk:
    """
    Halo model power spectrum P(k, z).
    """

    # FFTLog k grid for xi_1h/xi_2h -- a single class attribute k_grid
    k_grid = jnp.geomspace(1e-5, 1e3, 256)

    def __init__(self):
        # Define the P2xi object from mcfit, as we need to instantiate it before it can be used in a jitted function 
        self._p2xi = jax.jit(functools.partial(
            mcfit.P2xi(self.k_grid, lowring=True, backend='jax'),
            axis=0, extrap=False,
        ))

    # ------------------------------------------------------------------
    # 1-halo term
    # ------------------------------------------------------------------

    def pk_1h(self, halo_model, profile1, profile2, k, z, k_damp=0.01):
        """
        Compute the 1-halo contribution to the 3D power spectrum.

        .. math::

            P_{1h}(k, z) = \\int d\\ln M \\, \\frac{dn}{d\\ln M} \\, u_1(k, M, z) u_2(k, M, z)

        where :math:`dn/d\\ln M` is the halo mass function
        and :math:`u_i(k \\mid M, z)` is the Fourier-space tracer profile.
        The mass integral is performed over :attr:`m_grid`.

        Parameters
        ----------
        halo_model : HaloModel
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None
            Second halo profile object (if None, uses profile1).
        k : array-like
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.
        k_damp : float, default 0.01
            Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` for the low-k suppression factor.

        Returns
        -------
        pk_1h : array
            1-halo power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`, where singleton dimensions get squeezed before
            return.
        """
        hm = halo_model
        k, m, z = jnp.atleast_1d(k), hm.m_grid, jnp.atleast_1d(z)
        profile2 = profile2 if profile2 is not None else profile1

        # Weights and Setup
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        dndlnm = jnp.reshape(hm.halo_mass_function.dndlnm(hm.cosmology, m, z, hm.mass_def, hm.convert_masses), (len(m), len(z)))
        total_weights = dndlnm * w[:, None]  # (Nm, Nz)

        # Process a single mass bin at a time and extract the uk^2 at the lowest mass for the halo model consistency term
        def process_bin(i):
            pair_kernel = _fourier_2pt(hm, profile1, profile2, k, m, z)
            pair_kernel = jnp.reshape(pair_kernel, (len(k), len(m), len(z)))
            uk_sq_row = pair_kernel[:, i, :]

            return uk_sq_row * total_weights[i], uk_sq_row

        # vmap through the mass bins
        integrand_rows, all_sq_profiles = jax.vmap(process_bin)(jnp.arange(len(m)))

        pk1h = jnp.sum(integrand_rows, axis=0)

        # Apply halo model consistency correction: n_min * uk_sq_min
        uk_sq_min = all_sq_profiles[0]
        n_min, _, _ = hm._counter_terms(z)
        correction = n_min[None, :] * uk_sq_min
        pk1h = pk1h + hm.hm_consistency * correction

        # Apply damping
        mask = k_damp > 0
        damping = jnp.where(mask, 1.0 - jnp.exp(-(k / jnp.where(mask, k_damp, 1.0))**2), 1.0)

        return jnp.squeeze(pk1h * damping[:, None])

    # ------------------------------------------------------------------
    # 2-halo term
    # ------------------------------------------------------------------

    def pk_2h(self, halo_model, profile1, profile2, k, z):
        """
        Compute the 2-halo contribution to the 3D power spectrum.

        .. math::

            P_{2h}(k, z) = P_{\\mathrm{lin}}(k, z) \\, I_1(k, z) \\, I_2(k, z)

        with

        .. math::

            I_i(k, z) = \\int d\\ln M \\, \\frac{dn}{d\\ln M}(M, z) \\, b(M, z) \\, u_i(k \\mid M, z),

        where :math:`u_i(k \\mid M, z)` is the Fourier-space tracer profile,
        :math:`dn/d\\ln M` is the halo mass function, and :math:`b(M, z)` is the
        linear halo bias. The mass integral is performed over :attr:`m_grid`.

        Parameters
        ----------
        halo_model : HaloModel
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None
            Second halo profile object (if None, uses profile1).
        k : array-like
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.

        Returns
        -------
        pk_2h : array
            2-halo power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`, where singleton dimensions get squeezed before
            return.
        """
        hm = halo_model
        k, m, z = jnp.atleast_1d(k), hm.m_grid, jnp.atleast_1d(z)

        profile2 = profile2 if profile2 is not None else profile1

        # Weights and Ingredients
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        # Combine hmf, bias, and weights into a single (Nm, Nz) weight grid
        dndlnm = jnp.reshape(hm.halo_mass_function.dndlnm(hm.cosmology, m, z, hm.mass_def, hm.convert_masses), (len(m), len(z)))
        bias = jnp.reshape(hm.halo_bias.bias(hm.cosmology, m, z, hm.mass_def, hm.convert_masses), (len(m), len(z)))
        total_weights = dndlnm * bias * w[:, None]

        def get_I(profile):
            # This function processes a single index 'i' of the mass axis
            def process_bin(i):
                uk_full = jnp.reshape(profile.fourier(hm, k, m, z), (len(k), len(m), len(z)))
                uk_slice = uk_full[:, i, :]
                return uk_slice * total_weights[i], uk_slice

            # Vmap over the indices 0...Nm-1, then integrate and pluck index 0 for hm consistency
            integrand_rows, all_profiles = jax.vmap(process_bin)(jnp.arange(len(m)))
            integral = jnp.sum(integrand_rows, axis=0)
            u_k_min = all_profiles[0]  # vmap output is (Nm, Nk, Nz)

            n_min, b1_min, _ = hm._counter_terms(z)
            correction = b1_min[None, :] * n_min[None, :] * u_k_min

            return integral + hm.hm_consistency * correction

        # Final Power Spectrum
        I1 = get_I(profile1)
        I2 = I1 if profile1 is profile2 else get_I(profile2)

        P_lin = hm.cosmology.pk(k, z, linear=True)
        # Ensure P_lin has shape (N_k, N_z)
        P_lin = jnp.reshape(P_lin, (len(k), -1))

        return jnp.squeeze(P_lin * I1 * I2)

    # ------------------------------------------------------------------
    # Correlation function (FFTLog transform of pk_1h / pk_2h)
    # ------------------------------------------------------------------

    def xi_1h(self, halo_model, profile1, profile2, r, z, k_damp=0.01):
        """
        Compute the 1-halo contribution to the 3D correlation function.

        .. math::

            \\xi_{1h}(r, z) = \\frac{1}{2\\pi^2} \\int dk\\, k^2\\,
            P_{1h}(k, z)\\, j_0(kr)

        obtained by an FFTLog transform (:class:`mcfit.P2xi`) of
        :meth:`pk_1h`, tabulated on this :class:`Pk` instance's internal
        log-spaced ``k`` grid (:attr:`k_grid`) and interpolated onto the
        requested ``r``.

        Parameters
        ----------
        halo_model : HaloModel
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None
            Second halo profile object (if None, uses profile1).
        r : array-like
            Comoving separation grid in :math:`\\mathrm{Mpc}`. Only reliable
            well inside the range dual to :attr:`k_grid`; values of ``r`` too
            close to that range's edges are affected by FFTLog ringing.
        z : array-like
            Redshift grid.
        k_damp : float, default 0.01
            Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}`, passed through
            to :meth:`pk_1h`.

        Returns
        -------
        xi_1h : array
            1-halo correlation function (dimensionless), with shape
            :math:`(N_r, N_z)`, where singleton dimensions get squeezed
            before return.
        """
        r, z = jnp.atleast_1d(r), jnp.atleast_1d(z)

        pk = self.pk_1h(halo_model, profile1, profile2, self.k_grid, z, k_damp=k_damp)
        pk = jnp.reshape(pk, (len(self.k_grid), len(z)))

        r_native, xi_native = self._p2xi(pk)
        ln_r, ln_r_native = jnp.log(r), jnp.log(r_native)

        # Linear in xi against ln r rather than log-log
        def interp_col(xi_col):
            return jnp.interp(ln_r, ln_r_native, xi_col)

        xi = jax.vmap(interp_col, in_axes=1, out_axes=1)(xi_native)
        return jnp.squeeze(xi)

    def xi_2h(self, halo_model, profile1, profile2, r, z):
        """
        Compute the 2-halo contribution to the 3D correlation function.

        .. math::

            \\xi_{2h}(r, z) = \\frac{1}{2\\pi^2} \\int dk\\, k^2\\,
            P_{2h}(k, z)\\, j_0(kr)

        obtained by an FFTLog transform (:class:`mcfit.P2xi`) of
        :meth:`pk_2h`, tabulated on this :class:`Pk` instance's internal
        log-spaced ``k`` grid (:attr:`k_grid`) and interpolated onto the
        requested ``r``.

        Parameters
        ----------
        halo_model : HaloModel
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None
            Second halo profile object (if None, uses profile1).
        r : array-like
            Comoving separation grid in :math:`\\mathrm{Mpc}`. Only reliable
            well inside the range dual to :attr:`k_grid`; values of ``r`` too
            close to that range's edges are affected by FFTLog ringing.
        z : array-like
            Redshift grid.

        Returns
        -------
        xi_2h : array
            2-halo correlation function (dimensionless), with shape
            :math:`(N_r, N_z)`, where singleton dimensions get squeezed
            before return.
        """
        r, z = jnp.atleast_1d(r), jnp.atleast_1d(z)

        pk = self.pk_2h(halo_model, profile1, profile2, self.k_grid, z)
        pk = jnp.reshape(pk, (len(self.k_grid), len(z)))

        r_native, xi_native = self._p2xi(pk)
        ln_r, ln_r_native = jnp.log(r), jnp.log(r_native)

        # Linear in xi against ln r rather than log-log
        def interp_col(xi_col):
            return jnp.interp(ln_r, ln_r_native, xi_col)

        xi = jax.vmap(interp_col, in_axes=1, out_axes=1)(xi_native)
        return jnp.squeeze(xi)

    # ------------------------------------------------------------------
    # Angular power spectrum (Limber projection)
    # ------------------------------------------------------------------

    def cl_1h(self, halo_model, tracer1, tracer2, l, z, k_damp=0.01):
        """
        Compute the 1-halo contribution to the angular power spectrum
        :math:`C_\\ell^{1h}`.

        The Limber-projected spectrum is obtained by integrating the 1-halo
        3D power spectrum against the tracer kernels and the comoving volume
        element. The mass integral is performed over :attr:`m_grid`.

        Parameters
        ----------
        halo_model : HaloModel
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        l : array-like
            Multipole grid.
        z : array
            Redshift array. This must be an array because it defines the
            integration grid over redshift.
        k_damp : float, default 0.01
            Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` passed through to :meth:`pk_1h`.

        Returns
        -------
        cl_1h : array
            Dimensionless 1-halo angular power spectrum with shape
            :math:`(N_\\ell,)`, where singleton dimensions get squeezed before
            return.
        """
        hm = halo_model
        tracer2 = tracer1 if tracer2 is None else tracer2

        # Define the slice function to map l -> k for a specific z
        def get_pk_slice(zi):
            chi_i = hm.cosmology.angular_diameter_distance(zi) * (1 + zi)
            ki = (l + 0.5) / chi_i
            pk = self.pk_1h(hm, tracer1.profile, tracer2.profile, k=ki, z=jnp.atleast_1d(zi), k_damp=k_damp)
            return pk.flatten()

        # Get the halo model pk_1h, the kernels, and the Limber weight c/(H chi^2)
        P_1h_grid = jax.vmap(get_pk_slice)(z)
        kernel1 = tracer1.kernel(hm.cosmology, z)
        kernel2 = tracer2.kernel(hm.cosmology, z)
        chi = hm.cosmology.angular_diameter_distance(z) * (1.0 + z)
        limber_weight = hm.cosmology.comoving_volume_element(z) / chi**4

        # Limber integral: C_ell = int dz (c/H chi^2) W1 W2 P
        integrand = P_1h_grid * (limber_weight[:, None] * kernel1[:, None] * kernel2[:, None])

        return jnp.squeeze(jnp.trapezoid(integrand, x=z, axis=0))

    def cl_2h(self, halo_model, tracer1, tracer2, l, z):
        """
        Compute the 2-halo contribution to the angular power spectrum
        :math:`C_\\ell^{2h}`.

        The Limber-projected spectrum is obtained by integrating the 2-halo
        3D power spectrum against the tracer kernels and the comoving volume
        element. The mass integral is performed over :attr:`m_grid`.

        Parameters
        ----------
        halo_model : HaloModel
        tracer1 : Tracer
            First tracer object.
        tracer2 : Tracer or None
            Second tracer object (if None, uses tracer1).
        l : array-like
            Multipole grid.
        z : array
            Redshift array. This must be an array because it defines the
            integration grid over redshift.

        Returns
        -------
        cl_2h : array
            Dimensionless 2-halo angular power spectrum with shape
            :math:`(N_\\ell,)`, where singleton dimensions get squeezed before
            return.
        """
        hm = halo_model
        tracer2 = tracer1 if tracer2 is None else tracer2

        # Define the slice function for Limber integration
        def get_pk_slice(zi):
            # Map l to k using the Limber approximation and then get the pk_2h
            chi_i = hm.cosmology.angular_diameter_distance(zi) * (1 + zi)
            ki = (l + 0.5) / chi_i
            return self.pk_2h(hm, tracer1.profile, tracer2.profile, k=ki, z=jnp.atleast_1d(zi)).flatten()

        # Map over redshift to get P(k=l/chi, z)
        P_2h_grid = jax.vmap(get_pk_slice)(z)

        # Get individual kernels and the Limber weight c/(H chi^2)
        kernel1 = tracer1.kernel(hm.cosmology, z)
        kernel2 = tracer2.kernel(hm.cosmology, z)
        chi = hm.cosmology.angular_diameter_distance(z) * (1.0 + z)
        limber_weight = hm.cosmology.comoving_volume_element(z) / chi**4

        # Limber integral: C_ell = int dz (c/H chi^2) W1 W2 P
        integrand = P_2h_grid * (limber_weight[:, None] * kernel1[:, None] * kernel2[:, None])

        return jnp.squeeze(jnp.trapezoid(integrand, x=z, axis=0))


# -------------------------
# Halo model bispectrum
# -------------------------

class Bk:
    """
    Halo model bispectrum B(k1, k2, k3, z).

    For profiles whose higher moments reduce to simple products of their
    Fourier-space first moments (e.g. matter, tSZ, CMB lensing), the nth-order
    profile product within a single halo is taken as u1(k1,M)*...*un(kn,M).
    Profiles with more complex intra-halo occupancy statistics (HOD, CIB) will
    require a dedicated 3-point profile; this is left as a future extension point.
    """

    # ------------------------------------------------------------------
    # 1-halo term
    # ------------------------------------------------------------------

    def bk_1h(self, halo_model, profile1, profile2, profile3, k1, k2, k3, z, k_damp=0.01):
        """
        1-halo bispectrum term.

        .. math::

            B_{1h}(k_1, k_2, k_3, z) =
            \\int \\frac{dn}{d\\ln M}\\, u_1(k_1 \\mid M, z)\\,
            u_2(k_2 \\mid M, z)\\, u_3(k_3 \\mid M, z)\\, d\\ln M

        where :math:`u_i` are the Fourier-space profiles (first moments).
        For profiles with non-trivial intra-halo occupancy statistics, replace
        the triple product with a ``_fourier_3pt`` helper.

        A low-k suppression factor :math:`1 - e^{-(k_{\\min}/k_{\\mathrm{damp}})^2}`
        is applied at the smallest wavenumber of the triplet.

        Parameters
        ----------
        halo_model : HaloModel
        profile1, profile2, profile3 : HaloProfile
            The three halo profiles at wavenumbers k1, k2, k3 respectively.
        k1, k2, k3 : float
            Triangle wavenumbers in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.
        k_damp : float, default 0.01
            Damping wavenumber for the low-k suppression.

        Returns
        -------
        array
            1-halo bispectrum in :math:`\\mathrm{Mpc}^6`, squeezed.
        """
        hm = halo_model
        m, z_arr = hm.m_grid, jnp.atleast_1d(z)
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        dndlnm = jnp.reshape(
            hm.halo_mass_function.dndlnm(hm.cosmology, m, z_arr, hm.mass_def, hm.convert_masses),
            (len(m), len(z_arr)),
        )
        total_weights = dndlnm * w[:, None]  # (Nm, Nz)

        k1s, k2s, k3s = jnp.atleast_1d(k1), jnp.atleast_1d(k2), jnp.atleast_1d(k3)
        u1 = jnp.reshape(profile1.fourier(hm, k1s, m, z_arr), (len(k1s), len(m), len(z_arr)))
        u2 = jnp.reshape(profile2.fourier(hm, k2s, m, z_arr), (len(k2s), len(m), len(z_arr)))
        u3 = jnp.reshape(profile3.fourier(hm, k3s, m, z_arr), (len(k3s), len(m), len(z_arr)))

        triple = u1 * u2 * u3  # (1, Nm, Nz)
        bk1h = jnp.sum(triple * total_weights[None, :, :], axis=1)  # (1, Nz)

        n_min, _, _ = hm._counter_terms(z_arr)
        correction = n_min[None, :] * u1[:, 0, :] * u2[:, 0, :] * u3[:, 0, :]
        bk1h = bk1h + hm.hm_consistency * correction

        k_min = jnp.minimum(jnp.minimum(jnp.asarray(k1), jnp.asarray(k2)), jnp.asarray(k3))
        mask = k_damp > 0
        damping = jnp.where(mask, 1.0 - jnp.exp(-(k_min / jnp.where(mask, k_damp, 1.0))**2), 1.0)

        # Reshape damping to (N_k, 1) so it broadcasts correctly over (N_k, N_z)
        damping_bc = jnp.reshape(damping, jnp.shape(damping) + (1,))
        return jnp.squeeze(bk1h * damping_bc)

    # ------------------------------------------------------------------
    # 2-halo term
    # ------------------------------------------------------------------

    def bk_2h(self, halo_model, profile1, profile2, profile3, k1, k2, k3, z):
        """
        2-halo bispectrum term.

        .. math::

            B_{2h} = P_{\\mathrm{lin}}(k_1)\\, I^{(1)}_1(k_1)\\, J^{(1)}_{23}(k_2,k_3)
                   + P_{\\mathrm{lin}}(k_2)\\, I^{(1)}_2(k_2)\\, J^{(1)}_{13}(k_1,k_3)
                   + P_{\\mathrm{lin}}(k_3)\\, I^{(1)}_3(k_3)\\, J^{(1)}_{12}(k_1,k_2)

        where :math:`I^{(1)}_i = \\int dn/d\\ln M\\, b_1 u_i` and
        :math:`J^{(1)}_{ij} = \\int dn/d\\ln M\\, b_1 u_i u_j`.

        Parameters
        ----------
        halo_model : HaloModel
        profile1, profile2, profile3 : HaloProfile
        k1, k2, k3 : float
            Triangle wavenumbers in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.

        Returns
        -------
        array
            2-halo bispectrum in :math:`\\mathrm{Mpc}^6`, squeezed.
        """
        hm = halo_model
        z_arr = jnp.atleast_1d(z)

        I1 = hm._I(profile1, k1, z, bias_order=1)
        I2 = hm._I(profile2, k2, z, bias_order=1)
        I3 = hm._I(profile3, k3, z, bias_order=1)

        J23 = _pair_integral(hm, profile2, profile3, k2, k3, z)
        J13 = _pair_integral(hm, profile1, profile3, k1, k3, z)
        J12 = _pair_integral(hm, profile1, profile2, k1, k2, z)

        P1 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k1), z_arr, linear=True))
        P2 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k2), z_arr, linear=True))
        P3 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k3), z_arr, linear=True))

        return jnp.squeeze(P1 * I1 * J23 + P2 * I2 * J13 + P3 * I3 * J12)

    # ------------------------------------------------------------------
    # 3-halo term
    # ------------------------------------------------------------------

    def bk_3h(self, halo_model, profile1, profile2, profile3, k1, k2, k3, z):
        """
        3-halo bispectrum term.

        .. math::

            B_{3h} = B_{\\mathrm{tree}}(k_1,k_2,k_3)\\,
                     I^{(1)}_1 I^{(1)}_2 I^{(1)}_3
                   + I^{(2)}_1 I^{(1)}_2 I^{(1)}_3 P_2 P_3
                   + I^{(1)}_1 I^{(2)}_2 I^{(1)}_3 P_1 P_3
                   + I^{(1)}_1 I^{(1)}_2 I^{(2)}_3 P_1 P_2

        :math:`B_{\\mathrm{tree}}` uses the correct SPT F2 kernel
        (:math:`\\hat k_i \\cdot \\hat k_j = (k_3^2-k_1^2-k_2^2)/(2k_1k_2)`).

        Parameters
        ----------
        halo_model : HaloModel
        profile1, profile2, profile3 : HaloProfile
        k1, k2, k3 : float
            Triangle wavenumbers in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.

        Returns
        -------
        array
            3-halo bispectrum in :math:`\\mathrm{Mpc}^6`, squeezed.
        """
        hm = halo_model
        z_arr = jnp.atleast_1d(z)

        I1_b1 = hm._I(profile1, k1, z, bias_order=1)
        I2_b1 = hm._I(profile2, k2, z, bias_order=1)
        I3_b1 = hm._I(profile3, k3, z, bias_order=1)
        I1_b2 = hm._I(profile1, k1, z, bias_order=2)
        I2_b2 = hm._I(profile2, k2, z, bias_order=2)
        I3_b2 = hm._I(profile3, k3, z, bias_order=2)

        P1 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k1), z_arr, linear=True))
        P2 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k2), z_arr, linear=True))
        P3 = jnp.squeeze(hm.cosmology.pk(jnp.atleast_1d(k3), z_arr, linear=True))

        # Tree-level SPT bispectrum with correct cosine convention:
        # mu_ij = (k_k^2 - k_i^2 - k_j^2) / (2 k_i k_j)  [opposite-side law]
        k1a, k2a, k3a = jnp.asarray(k1), jnp.asarray(k2), jnp.asarray(k3)
        mu12 = _mu(k1a, k2a, k3a)
        mu23 = _mu(k2a, k3a, k1a)
        mu31 = _mu(k3a, k1a, k2a)
        B_tree = (
            2.0 * _F2(k1a, k2a, mu12) * P1 * P2
            + 2.0 * _F2(k2a, k3a, mu23) * P2 * P3
            + 2.0 * _F2(k3a, k1a, mu31) * P3 * P1
        )

        tree_term = B_tree * I1_b1 * I2_b1 * I3_b1

        # Quadratic-bias corrections 
        b2_term = (
            I1_b2 * I2_b1 * I3_b1 * P2 * P3
            + I1_b1 * I2_b2 * I3_b1 * P1 * P3
            + I1_b1 * I2_b1 * I3_b2 * P1 * P2
        )

        return jnp.squeeze(tree_term + b2_term)


# -------------------------
# Halo model trispectrum
# -------------------------

class Tk:
    """
    Halo model trispectrum T(k_u, k_v, z) in the parallelogram ("covariance")
    configuration k1 = -k2 = k_u, k3 = -k4 = k_v.

    Momentum conservation k1+k2+k3+k4=0 is automatically satisfied for any
    k_u, k_v and any relative angle between the two pairs, since each pair
    already sums to zero individually. This is the standard configuration
    entering the covariance of the power spectrum,
    Cov[P(k), P(k')] ⊃ T(k,-k,k',-k'), and (after angle-averaging over the
    residual relative orientation of the two pairs) it reduces the
    trispectrum to a function of exactly two wavenumbers, k_u and k_v.

    For profiles whose higher moments reduce to simple products of their
    Fourier-space first moments (e.g. matter, tSZ, CMB lensing), the nth-order
    profile product within a single halo is taken as u1(k1,M)*...*un(kn,M).
    Profiles with more complex intra-halo occupancy statistics (HOD, CIB) will
    require a dedicated 3-point and 4-point profile; this is left as a future extension point.
    """

    # ------------------------------------------------------------------
    # 1-halo term
    # ------------------------------------------------------------------

    def tk_1h(self, halo_model, profile1, profile2, profile3, profile4, k_u, k_v, z):
        """
        1-halo trispectrum term.

        .. math::

            T_{1h}(k_u, k_v, z) =
            \\int \\frac{dn}{d\\ln M}\\, u_1(k_u \\mid M, z)\\, u_2(k_u \\mid M, z)\\,
            u_3(k_v \\mid M, z)\\, u_4(k_v \\mid M, z)\\, d\\ln M

        where :math:`u_i` are the Fourier-space profiles (first moments).

        Parameters
        ----------
        halo_model : HaloModel
        profile1, profile2 : HaloProfile
            Profiles at wavenumber ``k_u`` (the :math:`u_1, u_2` pair).
        profile3, profile4 : HaloProfile
            Profiles at wavenumber ``k_v`` (the :math:`v_1, v_2` pair).
        k_u, k_v : float
            The two independent wavenumbers of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.

        Returns
        -------
        array
            1-halo trispectrum in :math:`\\mathrm{Mpc}^9`, squeezed.
        """
        hm = halo_model
        m, z_arr = hm.m_grid, jnp.atleast_1d(z)
        logm = jnp.log(m)
        dm = jnp.diff(logm)
        w = jnp.concatenate([jnp.array([dm[0]]), dm[:-1] + dm[1:], jnp.array([dm[-1]])]) * 0.5

        dndlnm = jnp.reshape(
            hm.halo_mass_function.dndlnm(hm.cosmology, m, z_arr, hm.mass_def, hm.convert_masses),
            (len(m), len(z_arr)),
        )
        total_weights = dndlnm * w[:, None]  # (Nm, Nz)

        k_us, k_vs = jnp.atleast_1d(k_u), jnp.atleast_1d(k_v)
        u1 = jnp.reshape(profile1.fourier(hm, k_us, m, z_arr), (len(k_us), len(m), len(z_arr)))
        u2 = jnp.reshape(profile2.fourier(hm, k_us, m, z_arr), (len(k_us), len(m), len(z_arr)))
        u3 = jnp.reshape(profile3.fourier(hm, k_vs, m, z_arr), (len(k_vs), len(m), len(z_arr)))
        u4 = jnp.reshape(profile4.fourier(hm, k_vs, m, z_arr), (len(k_vs), len(m), len(z_arr)))

        quad = u1 * u2 * u3 * u4  # (N, Nm, Nz)
        tk1h = jnp.sum(quad * total_weights[None, :, :], axis=1)  # (N, Nz)

        n_min, _, _ = hm._counter_terms(z_arr)
        correction = n_min[None, :] * u1[:, 0, :] * u2[:, 0, :] * u3[:, 0, :] * u4[:, 0, :]
        tk1h = tk1h + hm.hm_consistency * correction

        return jnp.squeeze(tk1h)

    # ------------------------------------------------------------------
    # 2-halo term
    # ------------------------------------------------------------------

    def _Pbar_kernel(self, hm, k, kp, z_arr):
        """
        Angle-averaged isotropized linear power spectrum
        :math:`\\bar P(k,k') = \\langle P_{\\mathrm{lin}}(|{\\bf k}+{\\bf k}'|)\\rangle_\\theta`
        entering the "22" diagram of the 2-halo trispectrum term.\
        """
        _, _, _, pkr = _kr_pkr(hm, k, kp, z_arr)
        wgt = _TRISPEC_THETA_WEIGHT[None, :]
        return jnp.sum(pkr * wgt, axis=1)

    def tk_2h(self, halo_model, profile1, profile2, profile3, profile4, k_u, k_v, z):
        """
        2-halo trispectrum term (sum of the "22" and "13" diagrams).

        .. math::

            T_{2h}^{(22)}(k_u,k_v,z) = \\bar P(k_u,k_v)\\,\\Big[
                J^{(1)}_{13}(k_u,k_v)\\, J^{(1)}_{24}(k_u,k_v)
              + J^{(1)}_{14}(k_u,k_v)\\, J^{(1)}_{32}(k_u,k_v) \\Big]

        where :math:`J^{(1)}_{ab}(k_u,k_v) = \\int dn/d\\ln M\\, b_1\\,
        u_a(k_u,M)\\, u_b(k_v,M)` (see ``_pair_integral``), and
        :math:`\\bar P` is the relative-angle average of
        :math:`P_{\\mathrm{lin}}(|{\\bf k}_u+{\\bf k}_v|)` (see
        ``_Pbar_kernel``) -- the third possible pairing vanishes because it
        would require :math:`P_{\\mathrm{lin}}(0) = 0`.

        .. math::

            T_{2h}^{(13)}(k_u,k_v,z) = P_{\\mathrm{lin}}(k_u)\\,\\Big[
                I^{(1)}_1(k_u)\\, K^{(1)}_{2;34}(k_u,k_v)
              + I^{(1)}_2(k_u)\\, K^{(1)}_{1;34}(k_u,k_v) \\Big]
              + (u_i \\leftrightarrow v_i)

        where :math:`K^{(1)}_{a;bc}(k_1,k_2) = \\int dn/d\\ln M\\, b_1\\,
        u_a(k_1,M)\\, u_b(k_2,M)\\, u_c(k_2,M)` (see ``_triple_integral``):
        one profile alone at :math:`k_1`, the other two together at
        :math:`k_2`.

        Parameters
        ----------
        halo_model : HaloModel
        profile1, profile2 : HaloProfile
            Profiles at wavenumber ``k_u``.
        profile3, profile4 : HaloProfile
            Profiles at wavenumber ``k_v``.
        k_u, k_v : float
            The two independent wavenumbers of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.

        Returns
        -------
        array
            2-halo trispectrum in :math:`\\mathrm{Mpc}^9`, squeezed.
        """
        hm = halo_model
        z_arr = jnp.atleast_1d(z)
        k_us, k_vs = jnp.atleast_1d(k_u), jnp.atleast_1d(k_v)

        # "22" diagram: two halos, each hosting a pair of legs.
        Pbar = self._Pbar_kernel(hm, k_us, k_vs, z_arr)
        pair_a = (
            _pair_integral(hm, profile1, profile3, k_u, k_v, z)
            * _pair_integral(hm, profile2, profile4, k_u, k_v, z)
        )
        pair_b = (
            _pair_integral(hm, profile1, profile4, k_u, k_v, z)
            * _pair_integral(hm, profile3, profile2, k_u, k_v, z)
        )
        tk_22 = Pbar * (pair_a + pair_b)

        # "13" diagram: one leg alone in a halo, the other three together.
        P_u = jnp.squeeze(hm.cosmology.pk(k_us, z_arr, linear=True))
        P_v = jnp.squeeze(hm.cosmology.pk(k_vs, z_arr, linear=True))

        I1u = hm._I(profile1, k_u, z, bias_order=1)
        I2u = hm._I(profile2, k_u, z, bias_order=1)
        I3v = hm._I(profile3, k_v, z, bias_order=1)
        I4v = hm._I(profile4, k_v, z, bias_order=1)

        K_u2 = _triple_integral(hm, profile2, profile3, profile4, k_u, k_v, z)
        K_u1 = _triple_integral(hm, profile1, profile3, profile4, k_u, k_v, z)
        K_v4 = _triple_integral(hm, profile4, profile1, profile2, k_v, k_u, z)
        K_v3 = _triple_integral(hm, profile3, profile1, profile2, k_v, k_u, z)

        tk_13 = P_u * (I1u * K_u2 + I2u * K_u1) + P_v * (I3v * K_v4 + I4v * K_v3)

        return jnp.squeeze(tk_22 + tk_13)

    # ------------------------------------------------------------------
    # 3-halo term
    # ------------------------------------------------------------------

    def _P3_kernel(self, hm, k, kp, z_arr):
        """
        Angle-averaged :math:`P_3(k,k') = \\langle P_{\\mathrm{lin}}(|{\\bf
        k}+{\\bf k}'|)\\, F_2(k,k',\\theta)\\rangle_\\theta` entering the
        tree-level bispectrum-type kernel of the 3-halo trispectrum term
        (``_Bpt_kernel``), built from the existing ``_F2``/``_mu`` SPT
        kernels evaluated between leg ``k`` and the vector sum
        ``kr = |k + kp|`` (``_ksum``).
        """
        k_b, kp_b, kr, pkr = _kr_pkr(hm, k, kp, z_arr)
        wgt = _TRISPEC_THETA_WEIGHT[None, :]
        f2_kkp = jnp.where(kr == 0.0, 13.0 / 28.0, _F2(k_b, kr, _mu(k_b, kr, kp_b)))
        return jnp.sum(pkr * f2_kkp * wgt, axis=1)

    def _Bpt_kernel(self, hm, k, kp, z_arr):
        """
        Tree-level, angle-averaged bispectrum-type kernel entering the
        3-halo trispectrum term, following Eq. 30 of Takada & Hu (2013):

        .. math::

            B^{\\mathrm{PT}}(k,k') = \\frac{12}{7} P_{\\mathrm{lin}}(k)
            P_{\\mathrm{lin}}(k') + 2\\big[P_{\\mathrm{lin}}(k)\\, P_3(k,k')
            + P_{\\mathrm{lin}}(k')\\, P_3(k',k)\\big]
        """
        P_k = jnp.squeeze(hm.cosmology.pk(k, z_arr, linear=True))
        P_kp = jnp.squeeze(hm.cosmology.pk(kp, z_arr, linear=True))
        P3_kkp = self._P3_kernel(hm, k, kp, z_arr)
        P3_kpk = self._P3_kernel(hm, kp, k, z_arr)
        return 12.0 / 7.0 * P_k * P_kp + 2.0 * (P_k * P3_kkp + P_kp * P3_kpk)

    def tk_3h(self, halo_model, profile1, profile2, profile3, profile4, k_u, k_v, z):
        """
        3-halo trispectrum term.

        .. math::

            T_{3h}(k_u,k_v,z) = B^{\\mathrm{PT}}(k_u,k_v)\\, \\Big[
                I^{(1)}_1(k_u) I^{(1)}_3(k_v) J^{(1)}_{24}(k_u,k_v)
              + I^{(1)}_1(k_u) I^{(1)}_4(k_v) J^{(1)}_{32}(k_u,k_v)
              + I^{(1)}_3(k_v) I^{(1)}_2(k_u) J^{(1)}_{14}(k_u,k_v)
              + I^{(1)}_4(k_v) I^{(1)}_2(k_u) J^{(1)}_{31}(k_u,k_v) \\Big]

        where :math:`B^{\\mathrm{PT}}` is the tree-level bispectrum-type
        kernel from ``_Bpt_kernel`` and :math:`J^{(1)}_{ab}` is the pair
        integral from ``_pair_integral``.

        Parameters
        ----------
        halo_model : HaloModel
        profile1, profile2 : HaloProfile
            Profiles at wavenumber ``k_u``.
        profile3, profile4 : HaloProfile
            Profiles at wavenumber ``k_v``.
        k_u, k_v : float
            The two independent wavenumbers of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.

        Returns
        -------
        array
            3-halo trispectrum in :math:`\\mathrm{Mpc}^9`, squeezed.
        """
        hm = halo_model
        z_arr = jnp.atleast_1d(z)
        k_us, k_vs = jnp.atleast_1d(k_u), jnp.atleast_1d(k_v)

        Bpt = self._Bpt_kernel(hm, k_us, k_vs, z_arr)

        I1u = hm._I(profile1, k_u, z, bias_order=1)
        I2u = hm._I(profile2, k_u, z, bias_order=1)
        I3v = hm._I(profile3, k_v, z, bias_order=1)
        I4v = hm._I(profile4, k_v, z, bias_order=1)

        J24 = _pair_integral(hm, profile2, profile4, k_u, k_v, z)
        J32 = _pair_integral(hm, profile3, profile2, k_u, k_v, z)
        J14 = _pair_integral(hm, profile1, profile4, k_u, k_v, z)
        J31 = _pair_integral(hm, profile3, profile1, k_u, k_v, z)

        tk3h = Bpt * (
            I1u * I3v * J24
            + I1u * I4v * J32
            + I3v * I2u * J14
            + I4v * I2u * J31
        )

        return jnp.squeeze(tk3h)

    # ------------------------------------------------------------------
    # 4-halo term
    # ------------------------------------------------------------------

    def _P4_kernel(self, hm, k, kp, z_arr):
        """
        Angle-averaged P4A(k,kp), P4X(k,kp) kernels ("1122" diagram) entering
        the tree-level 4h trispectrum, built from the existing ``_F2``/``_mu``
        SPT kernels evaluated between leg ``k`` and the vector sum
        ``kr = |k + kp|`` (``_ksum``), following Eq. 30 of Takada & Hu (2013).

        ``k``, ``kp`` : (N,) arrays — the two leg magnitudes for this ordering.
        Returns ``(P4A, P4X)``, each of shape (N, Nz).
        """
        k_b, kp_b, kr, pkr = _kr_pkr(hm, k, kp, z_arr)
        wgt = _TRISPEC_THETA_WEIGHT[None, :]

        # F2 between leg k (or kp) and the (negative of the) internal
        # propagator -kr, whose opposite side is the other leg (kp or k).
        f2_kkp = jnp.where(kr == 0.0, 13.0 / 28.0, _F2(k_b, kr, _mu(k_b, kr, kp_b)))
        f2_kpk = jnp.where(kr == 0.0, 13.0 / 28.0, _F2(kp_b, kr, _mu(kp_b, kr, k_b)))

        P4A = jnp.sum(pkr * f2_kkp ** 2 * wgt, axis=1)
        P4X = jnp.sum(pkr * f2_kkp * f2_kpk * wgt, axis=1)
        return P4A, P4X

    def tk_4h(self, halo_model, profile1, profile2, profile3, profile4, k_u, k_v, z):
        """
        4-halo (tree-level) trispectrum term.

        .. math::

            T_{4h}(k_u, k_v, z) = T^{\\mathrm{PT}}(k_u,-k_u,k_v,-k_v)\\,
            I^{(1)}_1(k_u)\\, I^{(1)}_2(k_u)\\, I^{(1)}_3(k_v)\\, I^{(1)}_4(k_v)

        where :math:`I^{(1)}_i = \\int dn/d\\ln M\\, b_1\\, u_i` (see
        ``HaloModel._I``) and the tree-level trispectrum :math:`T^{PT}` for
        the parallelogram configuration is angle-averaged over the relative
        orientation of the :math:`k_u` and :math:`k_v` pairs via a
        fixed-order Gauss-Legendre quadrature over :math:`\\theta \\in [0,\\pi]`
        (the "1122" diagram built from ``_F2``/``_mu``/``_ksum`` in
        ``_P4_kernel``; the "1113" diagram from the closed-form ``_X3``
        kernel), following Eq. 30 of Takada & Hu (2013).

        Parameters
        ----------
        halo_model : HaloModel
        profile1, profile2 : HaloProfile
            Profiles at wavenumber ``k_u``.
        profile3, profile4 : HaloProfile
            Profiles at wavenumber ``k_v``.
        k_u, k_v : float
            The two independent wavenumbers of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.

        Returns
        -------
        array
            4-halo trispectrum in :math:`\\mathrm{Mpc}^9`, squeezed.
        """
        hm = halo_model
        z_arr = jnp.atleast_1d(z)

        k_us, k_vs = jnp.atleast_1d(k_u), jnp.atleast_1d(k_v)

        P_u = jnp.squeeze(hm.cosmology.pk(k_us, z_arr, linear=True))
        P_v = jnp.squeeze(hm.cosmology.pk(k_vs, z_arr, linear=True))

        # "1113" diagram (tree-level F3-type kernel), symmetrized k_u <-> k_v
        X_uv = _X3(k_us, k_vs)
        X_vu = _X3(k_vs, k_us)
        t1113 = 4.0 / 9.0 * P_u ** 2 * P_v * X_uv + 4.0 / 9.0 * P_v ** 2 * P_u * X_vu

        # "1122" diagram (two F2-type vertices), symmetrized k_u <-> k_v
        P4A_uv, P4X_uv = self._P4_kernel(hm, k_us, k_vs, z_arr)
        P4A_vu, P4X_vu = self._P4_kernel(hm, k_vs, k_us, z_arr)
        t1122 = (
            8.0 * (P_u ** 2 * P4A_uv + P_u * P_v * P4X_uv)
            + 8.0 * (P_v ** 2 * P4A_vu + P_v * P_u * P4X_vu)
        )

        T_pt = t1113 + t1122

        I1 = hm._I(profile1, k_u, z, bias_order=1)
        I2 = hm._I(profile2, k_u, z, bias_order=1)
        I3 = hm._I(profile3, k_v, z, bias_order=1)
        I4 = hm._I(profile4, k_v, z, bias_order=1)

        return jnp.squeeze(T_pt * I1 * I2 * I3 * I4)


        