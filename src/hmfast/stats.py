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

    ``k``, ``kp`` : (Nk,), (Nkp,) independent arrays, broadcast into an
    (Nk, Nkp) grid of pairs. Returns an (Nk, Nkp) array, not squeezed --
    this is only ever used as an internal building block of ``Tk.tk_4h``.
    """
    r = kp[None, :] / k[:, None]              # (Nk, Nkp)
    cth = _TRISPEC_COS_THETA[None, None, :]   # (1, 1, N_theta)
    wgt = _TRISPEC_THETA_WEIGHT[None, None, :]

    r_b = r[:, :, None]  # (Nk, Nkp, 1)
    kr = _ksum(k[:, None, None], kp[None, :, None], cth)  # (Nk, Nkp, N_theta)
    intd = (
        (5.0 * r_b + (7.0 - 2.0 * r_b ** 2) * cth) / (1.0 + r_b ** 2 + 2.0 * r_b * cth)
        * (3.0 / 7.0 * r_b + 0.5 * (1.0 + r_b ** 2) * cth + 4.0 / 7.0 * r_b * cth ** 2)
    )
    intd = jnp.where(kr == 0.0, 0.0, intd)

    isotropized = jnp.sum(intd * wgt, axis=2)  # (Nk, Nkp)
    return -7.0 / 4.0 * (1.0 + r ** 2) + isotropized



# -------------------------
# Halo model mass-integral building blocks
# -------------------------
#
# Shared across Bk and Tk (not tied to either class's state), so kept at
# module level rather than duplicated as a private method on each.

def _pair_integral(halo_model, p1, p2, k1, k2, z, outer=False):
    """
    ∫ dn/dlnM * b1(M) * p1.fourier(k1,M,z) * p2.fourier(k2,M,z) dlnM

    Pair integral with first-order halo bias included.
    Used as a building block for ``Bk.bk_2h`` (``outer=False``: ``k1``
    and ``k2`` share a single batch axis, paired elementwise across a set
    of triangles) and the halo-model trispectrum's 2h/3h terms
    (``outer=True``: ``k1`` and ``k2`` are independent and broadcast into
    an (N1, N2) grid).

    Future generalisation: replace ``u1 * u2`` with a ``_fourier_2pt``
    variant that handles different k values when specialised 2-point kernels
    (HOD, CIB) are needed.

    Returns
    -------
    array
        ``outer=False``: shape (Nk, Nz), singleton dimensions squeezed.
        ``outer=True``: shape (N1, N2, Nz), not squeezed -- this branch is
        only ever used as an internal building block of ``Tk.tk_2h``/
        ``Tk.tk_3h``.
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

    n_min, b1_min, _ = hm._counter_terms(z_arr)

    if outer:
        u1e = u1[:, None, :, :]  # (N1, 1, Nm, Nz)
        u2e = u2[None, :, :, :]  # (1, N2, Nm, Nz)
        integral = jnp.sum(u1e * u2e * total_weights[None, None, :, :], axis=2)  # (N1, N2, Nz)
        correction = (
            n_min[None, None, :] * b1_min[None, None, :]
            * u1[:, None, 0, :] * u2[None, :, 0, :]
        )
        return integral + hm.hm_consistency * correction

    integral = jnp.sum(u1 * u2 * total_weights[None, :, :], axis=1)  # (Nk, Nz)
    correction = n_min[None, :] * b1_min[None, :] * u1[:, 0, :] * u2[:, 0, :]
    return jnp.squeeze(integral + hm.hm_consistency * correction)


def _triple_integral(halo_model, p_single, p_pair1, p_pair2, k_single, k_pair, z, outer=False):
    """
    ∫ dn/dlnM * b1(M) * p_single(k_single,M) * p_pair1(k_pair,M) * p_pair2(k_pair,M) dlnM

    Triple integral with first-order halo bias, generalising ``_pair_integral``
    to a "1x2" moment: one profile alone at ``k_single``, the other two both
    at ``k_pair``. Needed by the trispectrum's 2-halo "13" term.

    ``outer=False`` (default) pairs ``k_single``/``k_pair`` elementwise,
    sharing a single batch axis. ``outer=True`` broadcasts them into an
    independent (N_single, N_pair) grid, as needed by ``Tk.tk_2h``.

    Returns
    -------
    array
        ``outer=False``: shape (Nk, Nz), singleton dimensions squeezed.
        ``outer=True``: shape (N_single, N_pair, Nz), not squeezed -- this
        branch is only ever used as an internal building block of
        ``Tk.tk_2h``.
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

    n_min, b1_min, _ = hm._counter_terms(z_arr)

    if outer:
        u_se = u_s[:, None, :, :]            # (Ns, 1, Nm, Nz)
        u_pe = (u_p1 * u_p2)[None, :, :, :]  # (1, Np, Nm, Nz)
        integral = jnp.sum(u_se * u_pe * total_weights[None, None, :, :], axis=2)  # (Ns, Np, Nz)
        correction = (
            n_min[None, None, :] * b1_min[None, None, :]
            * u_s[:, None, 0, :] * u_p1[None, :, 0, :] * u_p2[None, :, 0, :]
        )
        return integral + hm.hm_consistency * correction

    integral = jnp.sum(u_s * u_p1 * u_p2 * total_weights[None, :, :], axis=1)  # (Nk, Nz)
    correction = n_min[None, :] * b1_min[None, :] * u_s[:, 0, :] * u_p1[:, 0, :] * u_p2[:, 0, :]
    return jnp.squeeze(integral + hm.hm_consistency * correction)


def _kr_pkr(hm, k, kp, z_arr):
    """
    Shared per-theta ``kr = |k + kp|`` and :math:`P_{\\mathrm{lin}}(k_r)`
    arrays entering every angle-averaged trispectrum kernel
    (``_Pbar_kernel``, ``_P3_kernel``, ``_P4_kernel``). Computing this once
    and reusing it avoids repeating the (relatively expensive) emulator
    evaluation of :math:`P_{\\mathrm{lin}}` across the 2h, 3h and 4h terms.

    ``k``, ``kp`` : (Nk,), (Nkp,) independent arrays, broadcast into an
    (Nk, Nkp) grid of pairs. Returns ``(k_b, kp_b, kr, pkr)`` with ``k_b``
    of shape (Nk,1,1), ``kp_b`` of shape (1,Nkp,1), ``kr`` of shape
    (Nk,Nkp,N_theta), and ``pkr`` of shape (Nk,Nkp,N_theta,Nz).
    """
    k_b, kp_b = k[:, None, None], kp[None, :, None]
    cth = _TRISPEC_COS_THETA[None, None, :]
    kr = _ksum(k_b, kp_b, cth)  # (Nk, Nkp, N_theta)
    nz = len(z_arr)
    pkr = jnp.reshape(
        hm.cosmology.pk(kr.flatten(), z_arr, linear=True),
        kr.shape + (nz,),
    )
    return k_b, kp_b, kr, pkr


# -------------------------
# Halo model power spectrum
# -------------------------

class Pk:
    """
    Halo model power spectrum.

    .. math::

        P(k, z) = P_{1h} + P_{2h}

    where the two terms are built from the generalised halo-model mass
    integral

    .. math::

        I_\\mu^{(\\beta)}(k_1, \\dots, k_\\mu, z) = \\int d\\ln M\\,
        \\frac{dn}{d\\ln M}\\, b_\\beta(M, z) \\prod_{i=1}^{\\mu} u_i(k_i \\mid M, z)

    where :math:`\\mu` is the number of profiles/wavenumbers in the
    product, :math:`b_\\beta` is the :math:`\\beta`-th order halo bias
    (:math:`b_0 = 1` unweighted, :math:`b_1` linear), and :math:`u_i` are
    the Fourier-space profiles (first moments). See :meth:`pk_1h` and
    :meth:`pk_2h` for how each term is assembled from
    :math:`I_\\mu^{(\\beta)}`.
    """

    # FFTLog k grid for xi_1h/xi_2h -- set as default in __init__
    k_grid = None

    def __init__(self, k_grid=None):
        # FFTLog k grid for xi_1h/xi_2h
        self.k_grid = k_grid if k_grid is not None else jnp.geomspace(1e-5, 1e3, 256)

        # Define the P2xi object from mcfit, as we need to instantiate it before it can be used in a jitted function
        self._p2xi = jax.jit(functools.partial(
            mcfit.P2xi(self.k_grid, lowring=True, backend='jax'),
            axis=0, extrap=False,
        ))

    # ------------------------------------------------------------------
    # 1-halo term
    # ------------------------------------------------------------------

    def pk_1h(self, halo_model, k, z, profile1, profile2=None, k_damp=0.01):
        """
        Compute the 1-halo contribution to the 3D power spectrum.

        .. math::

            P_{1h}(k, z) = I_2^{(0)}(k, k, z)

        where :math:`I_2^{(0)}` is the unweighted (:math:`\\beta=0`) pair
        mass integral :math:`I_\\mu^{(\\beta)}` with :math:`\\mu=2`,
        evaluated with both profiles at the same wavenumber :math:`k`.
        The mass integral is performed over :attr:`m_grid`.

        Parameters
        ----------
        halo_model : HaloModel
        k : array-like
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None, default None
            Second halo profile object. If None, defaults to profile1.
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

    def pk_2h(self, halo_model, k, z, profile1, profile2=None):
        """
        Compute the 2-halo contribution to the 3D power spectrum.

        .. math::

            P_{2h}(k, z) = P_{\\mathrm{lin}}(k, z) \\, I_1^{(1)}(k, z) \\, I_1^{(1)}(k, z)

        where :math:`I_1^{(1)}` is the linearly-biased (:math:`\\beta=1`)
        single-profile mass integral :math:`I_\\mu^{(\\beta)}` with
        :math:`\\mu=1`, evaluated once per profile at wavenumber
        :math:`k`. The mass integral is performed over :attr:`m_grid`.

        Parameters
        ----------
        halo_model : HaloModel
        k : array-like
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        z : array-like
            Redshift grid.
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None, default None
            Second halo profile object. If None, defaults to profile1.

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

    def xi_1h(self, halo_model, r, z, profile1, profile2=None, k_damp=0.01):
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
        r : array-like
            Comoving separation grid in :math:`\\mathrm{Mpc}`. Only reliable
            well inside the range dual to :attr:`k_grid`; values of ``r`` too
            close to that range's edges are affected by FFTLog ringing.
        z : array-like
            Redshift grid.
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None, default None
            Second halo profile object. If None, defaults to profile1.
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

        pk = self.pk_1h(halo_model, self.k_grid, z, profile1, profile2, k_damp=k_damp)
        pk = jnp.reshape(pk, (len(self.k_grid), len(z)))

        r_native, xi_native = self._p2xi(pk)
        ln_r, ln_r_native = jnp.log(r), jnp.log(r_native)

        # Linear in xi against ln r rather than log-log
        def interp_col(xi_col):
            return jnp.interp(ln_r, ln_r_native, xi_col)

        xi = jax.vmap(interp_col, in_axes=1, out_axes=1)(xi_native)
        return jnp.squeeze(xi)

    def xi_2h(self, halo_model, r, z, profile1, profile2=None):
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
        r : array-like
            Comoving separation grid in :math:`\\mathrm{Mpc}`. Only reliable
            well inside the range dual to :attr:`k_grid`; values of ``r`` too
            close to that range's edges are affected by FFTLog ringing.
        z : array-like
            Redshift grid.
        profile1 : HaloProfile
            First halo profile object.
        profile2 : HaloProfile or None, default None
            Second halo profile object. If None, defaults to profile1.

        Returns
        -------
        xi_2h : array
            2-halo correlation function (dimensionless), with shape
            :math:`(N_r, N_z)`, where singleton dimensions get squeezed
            before return.
        """
        r, z = jnp.atleast_1d(r), jnp.atleast_1d(z)

        pk = self.pk_2h(halo_model, self.k_grid, z, profile1, profile2)
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
            pk = self.pk_1h(hm, k=ki, z=jnp.atleast_1d(zi), profile1=tracer1.profile, profile2=tracer2.profile, k_damp=k_damp)
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
            return self.pk_2h(hm, k=ki, z=jnp.atleast_1d(zi), profile1=tracer1.profile, profile2=tracer2.profile).flatten()

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
    Halo model bispectrum.

    .. math::

        B(k_1, k_2, k_3, z) = B_{1h} + B_{2h} + B_{3h}

    where the three terms are built from the generalised halo-model mass
    integral

    .. math::

        I_\\mu^{(\\beta)}(k_1, \\dots, k_\\mu, z) = \\int d\\ln M\\,
        \\frac{dn}{d\\ln M}\\, b_\\beta(M, z) \\prod_{i=1}^{\\mu} u_i(k_i \\mid M, z)

    where :math:`\\mu` is the number of profiles/wavenumbers in the
    product, :math:`b_\\beta` is the :math:`\\beta`-th order halo bias
    (:math:`b_0 = 1` unweighted, :math:`b_1` linear, :math:`b_2`
    quadratic), and :math:`u_i` are the Fourier-space profiles (first
    moments). See :meth:`bk_1h`, :meth:`bk_2h` and :meth:`bk_3h` for how
    each term is assembled from :math:`I_\\mu^{(\\beta)}`.

    .. note::

        This implementation is limited to profiles whose 3-point function
        within a single halo reduces to the product of their (1-point)
        Fourier-space profiles, i.e. :math:`u_{123}(k_1,k_2,k_3 \\mid M) =
        u_1(k_1 \\mid M)\\, u_2(k_2 \\mid M)\\, u_3(k_3 \\mid M)`. This holds
        for matter density and electron pressure/density profiles, but
        not in general for profiles with non-trivial intra-halo occupancy
        statistics such as HOD or CIB.
    """

    # ------------------------------------------------------------------
    # 1-halo term
    # ------------------------------------------------------------------

    def bk_1h(self, halo_model, k1, k2, k3, z, profile1, profile2=None, profile3=None, k_damp=0.01):
        """
        1-halo bispectrum term.


        .. math::

            B_{1h}(k_1, k_2, k_3, z) = I_3^0(k_1, k_2, k_3, z)

        where :math:`I_3^0` is the unweighted (:math:`\\beta=0`) triple mass
        integral :math:`I_\\mu^{(\\beta)}` with :math:`\\mu=3`.

        A low-k suppression factor :math:`1 - e^{-(k_{\\min}/k_{\\mathrm{damp}})^2}`
        is applied at the smallest wavenumber of the triplet.

        Parameters
        ----------
        halo_model : HaloModel
        k1, k2, k3 : float or array-like
            Triangle wavenumbers in :math:`\\mathrm{Mpc}^{-1}`. Must all be the same size.
        z : array-like
            Redshift grid.
        profile1 : HaloProfile
            Halo profile at wavenumber k1.
        profile2, profile3 : HaloProfile or None, default None
            Halo profiles at wavenumbers k2, k3 respectively. Each defaults
            to profile1 if None.
        k_damp : float, default 0.01
            Damping wavenumber for the low-k suppression.

        Returns
        -------
        array
            1-halo bispectrum in :math:`\\mathrm{Mpc}^6`, shape :math:`(N_k, N_z)` before
            singleton dimensions get squeezed before return.
        """
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
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

    def bk_2h(self, halo_model, k1, k2, k3, z, profile1, profile2=None, profile3=None):
        """
        2-halo bispectrum term.

        .. math::

            B_{2h}(k_1, k_2, k_3, z) = P_{\\mathrm{lin}}(k_1)\\, I_1^1(k_1)\\, I_2^1(k_2, k_3)
            \\;+\\; 2\\,\\mathrm{cyc.}

        where :math:`I_1^1` and :math:`I_2^1` are the linearly-biased
        (:math:`\\beta=1`) single- and pair-profile mass integrals
        :math:`I_\\mu^{(\\beta)}`, and
        "+ 2 cyc" denotes the sum over the two cyclic permutations of
        :math:`(1,2,3)`.

        Parameters
        ----------
        halo_model : HaloModel
        k1, k2, k3 : float or array-like
            Triangle wavenumbers in :math:`\\mathrm{Mpc}^{-1}`. Must all be the same size.
        z : array-like
            Redshift grid.
        profile1 : HaloProfile
            Halo profile at wavenumber k1.
        profile2, profile3 : HaloProfile or None, default None
            Halo profiles at wavenumbers k2, k3 respectively. Each defaults
            to profile1 if None.

        Returns
        -------
        array
            2-halo bispectrum in :math:`\\mathrm{Mpc}^6`, shape :math:`(N_k, N_z)` before
            singleton dimensions get squeezed before return.
        """
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
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

    def bk_3h(self, halo_model, k1, k2, k3, z, profile1, profile2=None, profile3=None):
        """
        3-halo bispectrum term.

        .. math::

            \\begin{aligned}
                B_{3h}(k_1, k_2, k_3, z) &= B^{\\mathrm{PT}}(k_1, k_2, k_3)\\,
                I_1^1(k_1)\\, I_1^1(k_2)\\, I_1^1(k_3) \\\\
                &\\quad +\\; \\Big[\\, I_1^2(k_1)\\, I_1^1(k_2)\\, I_1^1(k_3)\\,
                P_{\\mathrm{lin}}(k_2)\\, P_{\\mathrm{lin}}(k_3) \\;+\\; \\mathrm{cyc} \\,\\Big]
            \\end{aligned}

        where :math:`I_1^\\beta(k_i)` is the single-profile mass integral
        :math:`I_\\mu^{(\\beta)}` (with
        :math:`\\mu=1`; :math:`\\beta=1` linear or :math:`\\beta=2`
        quadratic bias), "+ 2 cyc" denotes the sum over the two cyclic
        permutations of :math:`(1,2,3)`, and

        .. math::

            B^{\\mathrm{PT}}(k_1, k_2, k_3) = 2\\, F_2(k_1, k_2)\\, P_{\\mathrm{lin}}(k_1)\\,
            P_{\\mathrm{lin}}(k_2) \\;+\\; 2\\,\\mathrm{cyc.}

        is the tree-level SPT bispectrum and :math:`F_2` is the standard second-order SPT kernel.

        Parameters
        ----------
        halo_model : HaloModel
        k1, k2, k3 : float or array-like
            Triangle wavenumbers in :math:`\\mathrm{Mpc}^{-1}`. Must all be the same size.
        z : array-like
            Redshift grid.
        profile1 : HaloProfile
            Halo profile at wavenumber k1.
        profile2, profile3 : HaloProfile or None, default None
            Halo profiles at wavenumbers k2, k3 respectively. Each defaults
            to profile1 if None.

        Returns
        -------
        array
            3-halo bispectrum in :math:`\\mathrm{Mpc}^6`, shape :math:`(N_k, N_z)` before
            singleton dimensions get squeezed before return.
        """
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
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

    ``k_u`` and ``k_v`` need not be the same length: each term below
    broadcasts them into an independent (N_u, N_v) grid of pairs, one
    trispectrum value per (k_u, k_v) combination. Passing equal, identical
    arrays for both (``k_u = k_v = k``) gives the full (N_k, N_k) grid for
    that shared array.

    .. math::

        T(k_u, k_v, z) = T_{1h} + T_{2h} + T_{3h} + T_{4h}

    where the four terms are built from the generalised halo-model mass
    integral

    .. math::

        I_\\mu^{(\\beta)}(k_1, \\dots, k_\\mu, z) = \\int d\\ln M\\,
        \\frac{dn}{d\\ln M}\\, b_\\beta(M, z) \\prod_{i=1}^{\\mu} u_i(k_i \\mid M, z)

    where :math:`\\mu` is the number of
    profiles/wavenumbers in the product, :math:`b_\\beta` is the
    :math:`\\beta`-th order halo bias (:math:`b_0 = 1` unweighted,
    :math:`b_1` linear, :math:`b_2` quadratic), and :math:`u_i` are the
    Fourier-space profiles (first moments). A wavenumber argument repeated
    across several :math:`u_i` (e.g. :math:`I_3^1(k_i, k_j, k_j)`) feeds
    that value to more than one profile; where two profiles could share a
    wavenumber, a superscript :math:`(i)` on a wavenumber argument, e.g.
    :math:`k_u^{(i)}`, indicates that argument feeds profile :math:`i`'s
    Fourier transform :math:`u_i`. See :meth:`tk_1h`, :meth:`tk_2h`,
    :meth:`tk_3h` and :meth:`tk_4h` for how each term is assembled from
    :math:`I_\\mu^{(\\beta)}`.

    .. note::

        This implementation is limited to profiles whose 3- and 4-point
        functions within a single halo reduce to products of their
        (1-point) Fourier-space profiles, i.e.
        :math:`u_{1234}(k_1,k_2,k_3,k_4 \\mid M) = u_1(k_1 \\mid M)\\,
        u_2(k_2 \\mid M)\\, u_3(k_3 \\mid M)\\, u_4(k_4 \\mid M)` (and
        similarly for the 3-point sub-clumps entering the 2-halo "13" term).
        This holds for matter density and electron pressure/density profiles, 
        but not in general for profiles with non-trivial 
        intra-halo occupancy statistics such as HOD or CIB.
    """

    # ------------------------------------------------------------------
    # 1-halo term
    # ------------------------------------------------------------------

    def tk_1h(self, halo_model, k_u, k_v, z, profile1, profile2=None, profile3=None, profile4=None):
        """
        1-halo trispectrum term.

        .. math::

            T_{1h}(k_u, k_v, z) = I_4^0\\!\\left(k_u^{(1)}, k_u^{(2)},
            k_v^{(3)}, k_v^{(4)}\\right)

        where :math:`I_4^0` is the unweighted (:math:`\\beta=0`) quadruple
        mass integral :math:`I_\\mu^{(\\beta)}` with :math:`\\mu=4`.

        Parameters
        ----------
        halo_model : HaloModel
        k_u, k_v : float or array-like
            Independent wavenumber grids of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`. Need not be the
            same length -- the two are broadcast into an (N_u, N_v) grid,
            one trispectrum value per combination.
        z : array-like
            Redshift grid.
        profile1 : HaloProfile
            First profile at wavenumber ``k_u`` (the :math:`u_1` leg).
        profile2, profile3, profile4 : HaloProfile or None, default None
            Second profile at ``k_u`` (:math:`u_2`) and the two profiles at
            ``k_v`` (:math:`v_1, v_2`), respectively. If None, profile2 and
            profile3 default to profile1, and profile4 defaults to
            (the resolved) profile2.

        Returns
        -------
        array
            1-halo trispectrum in :math:`\\mathrm{Mpc}^9`, with shape
            :math:`(N_u, N_v, N_z)`, where singleton dimensions are
            squeezed before return.
        """
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
        profile4 = profile4 if profile4 is not None else profile2
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

        u12 = (u1 * u2)[:, None, :, :]  # (Nu, 1, Nm, Nz)
        u34 = (u3 * u4)[None, :, :, :]  # (1, Nv, Nm, Nz)
        tk1h = jnp.sum(u12 * u34 * total_weights[None, None, :, :], axis=2)  # (Nu, Nv, Nz)

        n_min, _, _ = hm._counter_terms(z_arr)
        correction = (
            n_min[None, None, :]
            * u1[:, None, 0, :] * u2[:, None, 0, :] * u3[None, :, 0, :] * u4[None, :, 0, :]
        )
        tk1h = tk1h + hm.hm_consistency * correction

        return jnp.squeeze(tk1h)

    # ------------------------------------------------------------------
    # 2-halo term
    # ------------------------------------------------------------------

    def _Pbar_kernel(self, hm, k, kp, z_arr):
        """
        Angle-averaged isotropized linear power spectrum
        :math:`\\bar P(k,k') = \\langle P_{\\mathrm{lin}}(|{\\bf k}+{\\bf k}'|)\\rangle_\\theta`
        entering the "22" diagram of the 2-halo trispectrum term.

        ``k``, ``kp`` : (Nk,), (Nkp,) independent arrays. Returns an
        (Nk, Nkp, Nz) array, not squeezed -- this is only ever used as an
        internal building block of ``tk_2h``.
        """
        _, _, _, pkr = _kr_pkr(hm, k, kp, z_arr)
        wgt = _TRISPEC_THETA_WEIGHT[None, None, :, None]
        return jnp.sum(pkr * wgt, axis=2)

    def tk_2h(self, halo_model, k_u, k_v, z, profile1, profile2=None, profile3=None, profile4=None):
        """
        2-halo trispectrum term (sum of the "22" and "13" diagrams).

        .. math::

            T_{2h}^{(22)}(k_u,k_v,z) = \\bar P(k_u,k_v)\\, \\Big[
                I_2^1\\!\\left(k_u^{(1)}, k_v^{(3)}\\right)\\,
                I_2^1\\!\\left(k_u^{(2)}, k_v^{(4)}\\right)
              + I_2^1\\!\\left(k_u^{(1)}, k_v^{(4)}\\right)\\,
                I_2^1\\!\\left(k_u^{(3)}, k_v^{(2)}\\right) \\Big]

        where :math:`I_2^1` is the linearly-biased (:math:`\\beta=1`)
        pair-profile mass integral :math:`I_\\mu^{(\\beta)}`, and
        :math:`\\bar P` is the relative-angle average
        of :math:`P_{\\mathrm{lin}}(|{\\bf k}_u+{\\bf k}_v|)` -- the third
        possible pairing vanishes because it would require
        :math:`P_{\\mathrm{lin}}(0) = 0`.

        .. math::

            \\begin{aligned}
                T_{2h}^{(13)}(k_u,k_v,z) &= P_{\\mathrm{lin}}(k_u)\\, \\Big[
                    I_1^1\\!\\left(k_u^{(1)}\\right)\\,
                    I_3^1\\!\\left(k_u^{(2)}, k_v^{(3)}, k_v^{(4)}\\right)
                    + (1 \\leftrightarrow 2) \\Big] \\\\
                &\\quad + P_{\\mathrm{lin}}(k_v)\\, \\Big[
                    I_1^1\\!\\left(k_v^{(3)}\\right)\\,
                    I_3^1\\!\\left(k_v^{(4)}, k_u^{(1)}, k_u^{(2)}\\right)
                    + (3 \\leftrightarrow 4) \\Big]
            \\end{aligned}

        where "+ (1 :math:`\\leftrightarrow` 2)" denotes the additional term
        obtained by swapping labels 1 and 2 in the preceding term, similarly
        for "+ (3 :math:`\\leftrightarrow` 4)", and :math:`I_1^1`,
        :math:`I_3^1` are the linearly-biased (:math:`\\beta=1`) single- and
        triple-profile mass integrals :math:`I_\\mu^{(\\beta)}` -- for
        :math:`I_3^1`, profile :math:`a` alone at :math:`k_i`, profiles
        :math:`b,c` together at :math:`k_j`.

        Parameters
        ----------
        halo_model : HaloModel
        k_u, k_v : float or array-like
            Independent wavenumber grids of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`. Need not be the
            same length -- the two are broadcast into an (N_u, N_v) grid,
            one trispectrum value per combination.
        z : array-like
            Redshift grid.
        profile1 : HaloProfile
            First profile at wavenumber ``k_u``.
        profile2, profile3, profile4 : HaloProfile or None, default None
            Second profile at ``k_u``, and the two profiles at ``k_v``,
            respectively. If None, profile2 and profile3 default to
            profile1, and profile4 defaults to (the resolved) profile2.

        Returns
        -------
        array
            2-halo trispectrum in :math:`\\mathrm{Mpc}^9`, with shape
            :math:`(N_u, N_v, N_z)`, where singleton dimensions are
            squeezed before return.
        """
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
        profile4 = profile4 if profile4 is not None else profile2
        z_arr = jnp.atleast_1d(z)
        k_us, k_vs = jnp.atleast_1d(k_u), jnp.atleast_1d(k_v)
        nu, nv, nz = len(k_us), len(k_vs), len(z_arr)

        # "22" diagram: two halos, each hosting a pair of legs.
        Pbar = self._Pbar_kernel(hm, k_us, k_vs, z_arr)  # (Nu, Nv, Nz)
        pair_a = (
            _pair_integral(hm, profile1, profile3, k_u, k_v, z, outer=True)
            * _pair_integral(hm, profile2, profile4, k_u, k_v, z, outer=True)
        )
        pair_b = (
            _pair_integral(hm, profile1, profile4, k_u, k_v, z, outer=True)
            * _pair_integral(hm, profile3, profile2, k_u, k_v, z, outer=True)
        )
        tk_22 = Pbar * (pair_a + pair_b)  # (Nu, Nv, Nz)

        # "13" diagram: one leg alone in a halo, the other three together.
        P_u = jnp.reshape(hm.cosmology.pk(k_us, z_arr, linear=True), (nu, 1, nz))
        P_v = jnp.reshape(hm.cosmology.pk(k_vs, z_arr, linear=True), (1, nv, nz))

        I1u = jnp.reshape(hm._I(profile1, k_u, z, bias_order=1), (nu, 1, nz))
        I2u = jnp.reshape(hm._I(profile2, k_u, z, bias_order=1), (nu, 1, nz))
        I3v = jnp.reshape(hm._I(profile3, k_v, z, bias_order=1), (1, nv, nz))
        I4v = jnp.reshape(hm._I(profile4, k_v, z, bias_order=1), (1, nv, nz))

        K_u2 = _triple_integral(hm, profile2, profile3, profile4, k_u, k_v, z, outer=True)
        K_u1 = _triple_integral(hm, profile1, profile3, profile4, k_u, k_v, z, outer=True)
        K_v4 = jnp.swapaxes(
            _triple_integral(hm, profile4, profile1, profile2, k_v, k_u, z, outer=True), 0, 1
        )
        K_v3 = jnp.swapaxes(
            _triple_integral(hm, profile3, profile1, profile2, k_v, k_u, z, outer=True), 0, 1
        )

        tk_13 = P_u * (I1u * K_u2 + I2u * K_u1) + P_v * (I3v * K_v4 + I4v * K_v3)  # (Nu, Nv, Nz)

        return jnp.squeeze(tk_22 + tk_13)

    # ------------------------------------------------------------------
    # 3-halo term
    # ------------------------------------------------------------------

    def _P3_kernel(self, hm, k, kp, z_arr):
        """
        Angle-averaged :math:`P_3(k,k') = \\langle P_{\\mathrm{lin}}(|{\\bf
        k}+{\\bf k}'|)\\, F_2(k,k',\\theta)\\rangle_\\theta` entering the
        tree-level bispectrum-type kernel of the 3-halo trispectrum term,
        built from the SPT kernels evaluated between leg ``k`` and the
        vector sum ``kr = |k + kp|``.

        ``k``, ``kp`` : (Nk,), (Nkp,) independent arrays. Returns an
        (Nk, Nkp, Nz) array, not squeezed -- this is only ever used as an
        internal building block of ``tk_3h``.
        """
        k_b, kp_b, kr, pkr = _kr_pkr(hm, k, kp, z_arr)
        wgt = _TRISPEC_THETA_WEIGHT[None, None, :]
        f2_kkp = jnp.where(kr == 0.0, 13.0 / 28.0, _F2(k_b, kr, _mu(k_b, kr, kp_b)))
        return jnp.sum(pkr * (f2_kkp * wgt)[..., None], axis=2)

    def _Bpt_kernel(self, hm, k, kp, z_arr):
        """
        Tree-level, angle-averaged bispectrum-type kernel entering the
        3-halo trispectrum term, following Eq. 30 of Takada & Hu (2013):

        .. math::

            B^{\\mathrm{PT}}(k,k') = \\frac{12}{7} P_{\\mathrm{lin}}(k)
            P_{\\mathrm{lin}}(k') + 2\\big[P_{\\mathrm{lin}}(k)\\, P_3(k,k')
            + P_{\\mathrm{lin}}(k')\\, P_3(k',k)\\big]

        ``k``, ``kp`` : (Nk,), (Nkp,) independent arrays. Returns an
        (Nk, Nkp, Nz) array, not squeezed -- this is only ever used as an
        internal building block of ``tk_3h``.
        """
        nk, nkp, nz = len(k), len(kp), len(z_arr)
        P_k = jnp.reshape(hm.cosmology.pk(k, z_arr, linear=True), (nk, 1, nz))
        P_kp = jnp.reshape(hm.cosmology.pk(kp, z_arr, linear=True), (1, nkp, nz))
        P3_kkp = self._P3_kernel(hm, k, kp, z_arr)
        P3_kpk = jnp.swapaxes(self._P3_kernel(hm, kp, k, z_arr), 0, 1)
        return 12.0 / 7.0 * P_k * P_kp + 2.0 * (P_k * P3_kkp + P_kp * P3_kpk)

    def tk_3h(self, halo_model, k_u, k_v, z, profile1, profile2=None, profile3=None, profile4=None):
        """
        3-halo trispectrum term.

        .. math::

            \\begin{aligned}
                T_{3h}(k_u,k_v,z) = B^{\\mathrm{PT}}(k_u,k_v)\\, \\Big[\\;
                & I_1^1\\!\\left(k_u^{(1)}\\right)\\, I_1^1\\!\\left(k_v^{(3)}\\right)\\,
                  I_2^1\\!\\left(k_u^{(2)}, k_v^{(4)}\\right) \\\\
                +\\;& I_1^1\\!\\left(k_u^{(1)}\\right)\\, I_1^1\\!\\left(k_v^{(4)}\\right)\\,
                  I_2^1\\!\\left(k_u^{(3)}, k_v^{(2)}\\right) \\\\
                +\\;& I_1^1\\!\\left(k_v^{(3)}\\right)\\, I_1^1\\!\\left(k_u^{(2)}\\right)\\,
                  I_2^1\\!\\left(k_u^{(1)}, k_v^{(4)}\\right) \\\\
                +\\;& I_1^1\\!\\left(k_v^{(4)}\\right)\\, I_1^1\\!\\left(k_u^{(2)}\\right)\\,
                  I_2^1\\!\\left(k_u^{(3)}, k_v^{(1)}\\right)\\; \\Big]
            \\end{aligned}

        where :math:`B^{\\mathrm{PT}}` is the tree-level bispectrum-type
        kernel, and :math:`I_1^1`, :math:`I_2^1` are the
        linearly-biased (:math:`\\beta=1`) single- and pair-profile mass
        integrals :math:`I_\\mu^{(\\beta)}`.

        Parameters
        ----------
        halo_model : HaloModel
        k_u, k_v : float or array-like
            Independent wavenumber grids of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`. Need not be the
            same length -- the two are broadcast into an (N_u, N_v) grid,
            one trispectrum value per combination.
        z : array-like
            Redshift grid.
        profile1 : HaloProfile
            First profile at wavenumber ``k_u``.
        profile2, profile3, profile4 : HaloProfile or None, default None
            Second profile at ``k_u``, and the two profiles at ``k_v``,
            respectively. If None, profile2 and profile3 default to
            profile1, and profile4 defaults to (the resolved) profile2.

        Returns
        -------
        array
            3-halo trispectrum in :math:`\\mathrm{Mpc}^9`, with shape
            :math:`(N_u, N_v, N_z)`, where singleton dimensions are
            squeezed before return.
        """
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
        profile4 = profile4 if profile4 is not None else profile2
        z_arr = jnp.atleast_1d(z)
        k_us, k_vs = jnp.atleast_1d(k_u), jnp.atleast_1d(k_v)
        nu, nv, nz = len(k_us), len(k_vs), len(z_arr)

        Bpt = self._Bpt_kernel(hm, k_us, k_vs, z_arr)  # (Nu, Nv, Nz)

        I1u = jnp.reshape(hm._I(profile1, k_u, z, bias_order=1), (nu, 1, nz))
        I2u = jnp.reshape(hm._I(profile2, k_u, z, bias_order=1), (nu, 1, nz))
        I3v = jnp.reshape(hm._I(profile3, k_v, z, bias_order=1), (1, nv, nz))
        I4v = jnp.reshape(hm._I(profile4, k_v, z, bias_order=1), (1, nv, nz))

        J24 = _pair_integral(hm, profile2, profile4, k_u, k_v, z, outer=True)
        J32 = _pair_integral(hm, profile3, profile2, k_u, k_v, z, outer=True)
        J14 = _pair_integral(hm, profile1, profile4, k_u, k_v, z, outer=True)
        J31 = _pair_integral(hm, profile3, profile1, k_u, k_v, z, outer=True)

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
        the tree-level 4h trispectrum, built from the SPT kernels evaluated
        between leg ``k`` and the vector sum ``kr = |k + kp|``, following
        Eq. 30 of Takada & Hu (2013).

        ``k``, ``kp`` : (Nk,), (Nkp,) independent arrays — the two leg
        magnitudes for this ordering. Returns ``(P4A, P4X)``, each of shape
        (Nk, Nkp, Nz), not squeezed -- this is only ever used as an
        internal building block of ``tk_4h``.
        """
        k_b, kp_b, kr, pkr = _kr_pkr(hm, k, kp, z_arr)
        wgt = _TRISPEC_THETA_WEIGHT[None, None, :]

        # F2 between leg k (or kp) and the (negative of the) internal
        # propagator -kr, whose opposite side is the other leg (kp or k).
        f2_kkp = jnp.where(kr == 0.0, 13.0 / 28.0, _F2(k_b, kr, _mu(k_b, kr, kp_b)))
        f2_kpk = jnp.where(kr == 0.0, 13.0 / 28.0, _F2(kp_b, kr, _mu(kp_b, kr, k_b)))

        P4A = jnp.sum(pkr * (f2_kkp ** 2 * wgt)[..., None], axis=2)
        P4X = jnp.sum(pkr * (f2_kkp * f2_kpk * wgt)[..., None], axis=2)
        return P4A, P4X

    def tk_4h(self, halo_model, k_u, k_v, z, profile1, profile2=None, profile3=None, profile4=None):
        """
        4-halo (tree-level) trispectrum term.

        .. math::

            T_{4h}(k_u, k_v, z) = T^{\\mathrm{PT}}(k_u,-k_u,k_v,-k_v)\\,
            I_1^1\\!\\left(k_u^{(1)}\\right)\\, I_1^1\\!\\left(k_u^{(2)}\\right)\\,
            I_1^1\\!\\left(k_v^{(3)}\\right)\\, I_1^1\\!\\left(k_v^{(4)}\\right)

        where :math:`I_1^1` is the linearly-biased (:math:`\\beta=1`)
        single-profile mass integral :math:`I_\\mu^{(\\beta)}` and the
        tree-level trispectrum :math:`T^{\\mathrm{PT}}` for the
        parallelogram configuration is
        angle-averaged over the relative
        orientation of the :math:`k_u` and :math:`k_v` pairs via a
        fixed-order Gauss-Legendre quadrature over :math:`\\theta \\in [0,\\pi]`
        (the "1122" diagram built from the SPT kernels; the "1113" diagram
        from a closed-form kernel), following Eq. 30 of Takada & Hu (2013).

        Parameters
        ----------
        halo_model : HaloModel
        k_u, k_v : float or array-like
            Independent wavenumber grids of the parallelogram
            configuration, in :math:`\\mathrm{Mpc}^{-1}`. Need not be the
            same length -- the two are broadcast into an (N_u, N_v) grid,
            one trispectrum value per combination.
        z : array-like
            Redshift grid.
        profile1 : HaloProfile
            First profile at wavenumber ``k_u``.
        profile2, profile3, profile4 : HaloProfile or None, default None
            Second profile at ``k_u``, and the two profiles at ``k_v``,
            respectively. If None, profile2 and profile3 default to
            profile1, and profile4 defaults to (the resolved) profile2.

        Returns
        -------
        array
            4-halo trispectrum in :math:`\\mathrm{Mpc}^9`, with shape
            :math:`(N_u, N_v, N_z)`, where singleton dimensions are
            squeezed before return.
        """
        hm = halo_model
        profile2 = profile2 if profile2 is not None else profile1
        profile3 = profile3 if profile3 is not None else profile1
        profile4 = profile4 if profile4 is not None else profile2
        z_arr = jnp.atleast_1d(z)

        k_us, k_vs = jnp.atleast_1d(k_u), jnp.atleast_1d(k_v)
        nu, nv, nz = len(k_us), len(k_vs), len(z_arr)

        P_u = jnp.reshape(hm.cosmology.pk(k_us, z_arr, linear=True), (nu, 1, nz))
        P_v = jnp.reshape(hm.cosmology.pk(k_vs, z_arr, linear=True), (1, nv, nz))

        # "1113" diagram (tree-level F3-type kernel), symmetrized k_u <-> k_v
        X_uv = _X3(k_us, k_vs)[:, :, None]                            # (Nu, Nv, 1)
        X_vu = jnp.swapaxes(_X3(k_vs, k_us), 0, 1)[:, :, None]        # (Nu, Nv, 1)
        t1113 = 4.0 / 9.0 * P_u ** 2 * P_v * X_uv + 4.0 / 9.0 * P_v ** 2 * P_u * X_vu

        # "1122" diagram (two F2-type vertices), symmetrized k_u <-> k_v
        P4A_uv, P4X_uv = self._P4_kernel(hm, k_us, k_vs, z_arr)  # (Nu, Nv, Nz)
        P4A_vu, P4X_vu = self._P4_kernel(hm, k_vs, k_us, z_arr)  # (Nv, Nu, Nz)
        P4A_vu, P4X_vu = jnp.swapaxes(P4A_vu, 0, 1), jnp.swapaxes(P4X_vu, 0, 1)
        t1122 = (
            8.0 * (P_u ** 2 * P4A_uv + P_u * P_v * P4X_uv)
            + 8.0 * (P_v ** 2 * P4A_vu + P_v * P_u * P4X_vu)
        )

        T_pt = t1113 + t1122  # (Nu, Nv, Nz)

        I1 = jnp.reshape(hm._I(profile1, k_u, z, bias_order=1), (nu, 1, nz))
        I2 = jnp.reshape(hm._I(profile2, k_u, z, bias_order=1), (nu, 1, nz))
        I3 = jnp.reshape(hm._I(profile3, k_v, z, bias_order=1), (1, nv, nz))
        I4 = jnp.reshape(hm._I(profile4, k_v, z, bias_order=1), (1, nv, nz))

        return jnp.squeeze(T_pt * I1 * I2 * I3 * I4)


        