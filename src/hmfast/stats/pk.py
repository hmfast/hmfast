import functools

import jax
import jax.numpy as jnp
import mcfit

from hmfast.halos.profiles.profiles_2pt import _fourier_2pt
from .nonlimber import _cl_2h_nonlimber


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

        I_\\mu^\\beta(k_1, \\dots, k_\\mu, z) = \\int d\\ln M\\,
        \\frac{dn}{d\\ln M}\\, b_\\beta(M, z) \\prod_{i=1}^{\\mu} u_i(k_i \\,|\\, M, z)

    where :math:`\\mu` is the number of profiles/wavenumbers in the
    product, :math:`b_\\beta` is the :math:`\\beta`-th order halo bias
    (:math:`b_0 = 1` unweighted, :math:`b_1` linear), and :math:`u_i` are
    the Fourier-space profiles (first moments). See :meth:`pk_1h` and
    :meth:`pk_2h` for how each term is assembled from
    :math:`I_\\mu^\\beta`.
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

            P_{1h}(k, z) = I_2^0(k, k, z)

        where :math:`I_2^0` is the unweighted (:math:`\\beta=0`) pair
        mass integral :math:`I_\\mu^\\beta` with :math:`\\mu=2`,
        evaluated with both profiles at the same wavenumber :math:`k`.
        The mass integral is performed over :attr:`m_grid`.

        Parameters
        ----------
        halo_model : HaloModel
        k : float or jnp.ndarray
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        z : float or jnp.ndarray
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

        dndlnm = jnp.reshape(hm.halo_mass_function.dndlnm(hm.cosmology, m, z, hm.mass_def), (len(m), len(z)))
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

            P_{2h}(k, z) = P_{\\mathrm{lin}}(k, z) \\, I_1^1(k, z) \\, I_1^1(k, z)

        where :math:`I_1^1` is the linearly-biased (:math:`\\beta=1`)
        single-profile mass integral :math:`I_\\mu^\\beta` with
        :math:`\\mu=1`, evaluated once per profile at wavenumber
        :math:`k`. The mass integral is performed over :attr:`m_grid`.

        Parameters
        ----------
        halo_model : HaloModel
        k : float or jnp.ndarray
            Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`.
        z : float or jnp.ndarray
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
        dndlnm = jnp.reshape(hm.halo_mass_function.dndlnm(hm.cosmology, m, z, hm.mass_def), (len(m), len(z)))
        bias = jnp.reshape(hm.halo_bias.bias(hm.cosmology, m, z, hm.mass_def), (len(m), len(z)))
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
        r : float or jnp.ndarray
            Comoving separation grid in :math:`\\mathrm{Mpc}`. Only reliable
            well inside the range dual to :attr:`k_grid`; values of ``r`` too
            close to that range's edges are affected by FFTLog ringing.
        z : float or jnp.ndarray
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
        r : float or jnp.ndarray
            Comoving separation grid in :math:`\\mathrm{Mpc}`. Only reliable
            well inside the range dual to :attr:`k_grid`; values of ``r`` too
            close to that range's edges are affected by FFTLog ringing.
        z : float or jnp.ndarray
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

    @functools.partial(jax.jit, static_argnums=(0,), static_argnames=("include_1h", "include_2h"))
    def _cl_limber(self, halo_model, tracer1, tracer2, l, z, include_1h=False, include_2h=True, k_damp=0.01):
        """
        Limber C_ell for either or both halo terms; shared implementation
        behind :meth:`cl_1h` (``include_1h=True, include_2h=False``),
        :meth:`cl_2h` (the defaults), and the high-ell branch of
        :meth:`cl_2h_nonlimber`. ``l`` may be traced. Jitted with ``self``
        static (``Pk`` isn't a registered JAX pytree), so repeated calls on
        the *same* ``Pk`` instance reuse the cached compilation.
        """
        hm = halo_model
        cosmology = hm.cosmology
        tracer2 = tracer1 if tracer2 is None else tracer2
        z = jnp.atleast_1d(z)
        z_b = cosmology._z_grid_pk()[-1]

        # growth_factor is evaluated on the full z array, not per-scalar inside vmap, to match pk_1h/pk_2h exactly.
        in_bounds = z <= z_b
        growth_ratio_sq = jnp.where(in_bounds, 1.0, (cosmology.growth_factor(z) / cosmology.growth_factor(z_b)) ** 2)
        z_eval = jnp.where(in_bounds, z, z_b)

        def get_pk_slice(zi, zi_eval):
            chi_i = cosmology.angular_diameter_distance(zi) * (1.0 + zi)
            ki = (l + 0.5) / chi_i
            zi_eval = jnp.atleast_1d(zi_eval)

            p = 0.0
            if include_1h:
                p = p + self.pk_1h(hm, ki, zi_eval, tracer1.profile, tracer2.profile, k_damp=k_damp)
            if include_2h:
                p = p + self.pk_2h(hm, ki, zi_eval, tracer1.profile, tracer2.profile)
            return jnp.atleast_1d(p).flatten()

        P_grid = jax.vmap(get_pk_slice)(z, z_eval) * growth_ratio_sq[:, None]
        kernel1 = jnp.atleast_1d(tracer1.kernel(cosmology, z))
        kernel2 = jnp.atleast_1d(tracer2.kernel(cosmology, z))
        chi = cosmology.angular_diameter_distance(z) * (1.0 + z)
        limber_weight = cosmology.comoving_volume_element(z) / chi**4

        integrand = P_grid * (limber_weight[:, None] * kernel1[:, None] * kernel2[:, None])
        return jnp.squeeze(jnp.trapezoid(integrand, x=z, axis=0))

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
        l : float or jnp.ndarray
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
        return self._cl_limber(halo_model, tracer1, tracer2, l, z,
                                include_1h=True, include_2h=False, k_damp=k_damp)

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
        l : float or jnp.ndarray
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
        return self._cl_limber(halo_model, tracer1, tracer2, l, z, include_2h=True)

    # ------------------------------------------------------------------
    # Non-Limber angular power spectrum (2-halo term; experimental)
    # ------------------------------------------------------------------

    def cl_2h_nonlimber(self, halo_model, tracer1, tracer2, l, z, l_limber=0.0,
                         k=None, z_fid=0.0, n_chi=None, n_interp=200, bias=0.1, window=0.2):
        """
        2-halo angular power spectrum C_ell, exact at low ell and via the
        fast Limber approximation (:meth:`cl_2h`) at high ell.

        .. note::

            Experimental: this method runs alongside :meth:`cl_2h` while
            it's being validated against Limber-only results, and is
            expected to eventually replace it.

        The Limber approximation assumes each tracer's kernel varies slowly
        compared to the density fluctuations it's weighted against; this
        breaks down at low multipoles, where kernels can be narrow or only
        partially overlap in redshift. Below `l_limber`, this method
        instead performs an exact projection via a closed-form
        Hankel-transform (FFTLog) decomposition of each kernel
        (:func:`~hmfast.stats.nonlimber._cl_2h_nonlimber`); at and above
        `l_limber`, it falls back to :meth:`cl_2h`. Only the 2-halo
        contribution is computed here; add a 1-halo term separately (e.g.
        from :meth:`cl_1h`) for the full spectrum.

        This method itself is plain Python, not ``jax.jit``-compiled: which
        ell values go to which method is a data-dependent shape decision no
        array library can trace, so `l` must be concrete here (not a value
        being traced by JAX) and the low/high split is done as ordinary
        Python index bookkeeping. The two branches it dispatches to,
        :func:`~hmfast.stats.nonlimber._cl_2h_nonlimber` and
        :meth:`_cl_limber`, are each independently ``jax.jit``-compiled and
        cached by their own input shapes and static config
        (``n_chi``/``n_interp``/``bias``/``window`` for the former,
        ``include_1h``/``include_2h``/``self`` for the latter) -- so
        repeated calls on the *same* ``Pk`` instance with the same low/high
        ell *counts* reuse the cached compilation even when the concrete
        ell values, `z`, or any parameter reachable through
        `halo_model`/`tracer1`/`tracer2` change, and those all remain fully
        traced and differentiable through the cached functions.

        Parameters
        ----------
        halo_model : HaloModel
            Assembled halo model (cosmology, mass function, bias, concentration).
        tracer1 : Tracer
            First tracer.
        tracer2 : Tracer or None
            Second tracer; if None, computes the auto-correlation of tracer1.
        l : array-like
            Multipole values. Must be concrete (not a value being traced by
            JAX), since choosing which method to use per multipole is a
            data-dependent shape decision JAX can never trace (with any
            array library).
        z : array-like
            Redshift sampling. Only its minimum, maximum, and length are
            used, to set up internal grids; each tracer's own redshift
            support is covered automatically even if narrower than `z`, so
            it's safe to reuse whatever `z` grid you'd use elsewhere for
            these tracers.
        l_limber : float, default 0.0
            Multipoles below this use the exact low-ell method; multipoles
            at or above it use the Limber approximation. The default, 0.0,
            uses Limber everywhere; raise it to switch to the exact method
            at low ell, where it matters most.
        k : array-like or None, default None
            Wavenumber grid used internally by the exact low-ell
            calculation. Defaults to the cosmology's own tabulated
            power-spectrum grid.
        z_fid : float, default 0.0
            Fiducial redshift used to normalize the exact low-ell
            calculation's internal model of how the power spectrum evolves
            with redshift.
        n_chi : int or None, default None
            Resolution of the internal comoving-distance grid used by the
            exact low-ell calculation. Defaults to ``len(z)``.
        n_interp : int, default 200
            Number of coarse wavenumber points at which the exact low-ell
            calculation evaluates its most expensive step, before
            interpolating onto the full resolution set by `k`.
        bias : float, default 0.1
            Technical exponent (must be less than 1) controlling how the
            exact low-ell calculation numerically decomposes each kernel;
            the default works well across tracer types and rarely needs
            changing.
        window : float, default 0.2
            Fraction of FFTLog (Mellin) modes, at the high-frequency end,
            smoothly anti-aliased to suppress edge/periodicity ringing --
            matches the windowing SwiftCl's own FFTLog backend applies.
            Unlike a real-space taper, this doesn't discard genuine kernel
            amplitude near the domain edge, so it works uniformly across
            tracer types.

        Returns
        -------
        cl_2h : array, shape (N_ell,)
            The 2-halo angular power spectrum, in the same order as `l`.

        Notes
        -----
        Gradients via ``jax.grad``/``jax.jacobian`` work with respect to `z`
        and any parameter reachable through `halo_model`/`tracer1`/
        `tracer2`. `l`/`l_limber` are concrete Python-level dispatch
        inputs, not part of either branch's jit cache key -- only the
        resulting low/high ell *counts* matter for recompilation.
        `n_chi`/`n_interp`/`bias`/`window` remain static (they drive shapes
        or dispatch, not physics) and re-trigger a (cached-per-value)
        recompilation of the affected branch if changed.
        """
        tracer2 = tracer1 if tracer2 is None else tracer2

        l_arr = jnp.atleast_1d(jnp.asarray(l, dtype=jnp.float64))
        l_vals = [float(x) for x in l_arr]
        idx_low = [i for i, li in enumerate(l_vals) if li < l_limber]
        idx_high = [i for i, li in enumerate(l_vals) if li >= l_limber]

        result = jnp.zeros(len(l_vals), dtype=jnp.float64)
        if idx_low:
            low_idx = jnp.array(idx_low)
            result = result.at[low_idx].set(jnp.atleast_1d(
                _cl_2h_nonlimber(halo_model, tracer1, tracer2, l_arr[low_idx], z,
                                 k=k, z_fid=z_fid, n_chi=n_chi, n_interp=n_interp, bias=bias, window=window)))
        if idx_high:
            high_idx = jnp.array(idx_high)
            result = result.at[high_idx].set(jnp.atleast_1d(
                self._cl_limber(halo_model, tracer1, tracer2, l_arr[high_idx], z, include_2h=True)))
        return jnp.squeeze(result)
