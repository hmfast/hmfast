import jax
import jax.numpy as jnp


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

    Attributes
    ----------
    include_1h : bool
        Whether :meth:`pk_tot` includes the 1-halo term.
    include_2h : bool
        Whether :meth:`pk_tot` includes the 2-halo term.
    k_damp : float
        Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` for :meth:`pk_1h`'s
        low-k suppression factor.
    alpha_smooth : float
        HMcode-style exponent smoothing the transition between the 1-halo
        and 2-halo regimes in :meth:`pk_tot`; ``1.0`` recovers a plain sum.
    """

    def __init__(self, include_1h=True, include_2h=True, k_damp=0.01, alpha_smooth=1.0):
        """
        Parameters
        ----------
        include_1h : bool, default True
            Whether :meth:`pk_tot` includes the 1-halo term.
        include_2h : bool, default True
            Whether :meth:`pk_tot` includes the 2-halo term.
        k_damp : float, default 0.01
            Damping wavenumber in :math:`\\mathrm{Mpc}^{-1}` for :meth:`pk_1h`'s
            low-k suppression factor.
        alpha_smooth : float, default 1.0
            HMcode-style exponent smoothing the transition between the
            1-halo and 2-halo regimes in :meth:`pk_tot`; ``1.0`` recovers a
            plain sum.
        """
        self.include_1h = include_1h
        self.include_2h = include_2h
        self.k_damp = jnp.asarray(k_damp)
        self.alpha_smooth = jnp.asarray(alpha_smooth)

    def _tree_flatten(self):
        return (self.k_damp, self.alpha_smooth), (self.include_1h, self.include_2h)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.k_damp, obj.alpha_smooth = children
        obj.include_1h, obj.include_2h = aux_data
        return obj

    # ------------------------------------------------------------------
    # 1-halo term
    # ------------------------------------------------------------------

    @jax.jit
    def pk_1h(self, halo_model, k, z, profile1, profile2=None):
        """
        Compute the 1-halo contribution to the 3D power spectrum.

        .. math::

            P_{1h}(k, z) = I_2^0(k, k, z)

        where :math:`I_2^0` is the unweighted (:math:`\\beta=0`) pair
        mass integral :math:`I_\\mu^\\beta` with :math:`\\mu=2`,
        evaluated with both profiles at the same wavenumber :math:`k`.
        The mass integral is performed over :attr:`m_range`/:attr:`n_m`.

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
        pk_1h : array
            1-halo power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`, where singleton dimensions get squeezed before
            return.
        """
        hm = halo_model
        k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)
        profile2 = profile2 if profile2 is not None else profile1

        pk1h = jnp.reshape(hm.mass_integral(k, z, (profile1, profile2), bias_order=0), (len(k), len(z)))

        # Apply damping
        mask = self.k_damp > 0
        damping = jnp.where(mask, 1.0 - jnp.exp(-(k / jnp.where(mask, self.k_damp, 1.0))**2), 1.0)

        return jnp.squeeze(pk1h * damping[:, None])

    # ------------------------------------------------------------------
    # 2-halo term
    # ------------------------------------------------------------------

    @jax.jit
    def pk_2h(self, halo_model, k, z, profile1, profile2=None):
        """
        Compute the 2-halo contribution to the 3D power spectrum.

        .. math::

            P_{2h}(k, z) = P_{\\mathrm{lin}}(k, z) \\, I_1^1(k, z) \\, I_1^1(k, z)

        where :math:`I_1^1` is the linearly-biased (:math:`\\beta=1`)
        single-profile mass integral :math:`I_\\mu^\\beta` with
        :math:`\\mu=1`, evaluated once per profile at wavenumber
        :math:`k`. The mass integral is performed over :attr:`m_range`/:attr:`n_m`.

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
        k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)

        profile2 = profile2 if profile2 is not None else profile1

        I1 = jnp.reshape(hm.mass_integral(k, z, profile1, bias_order=1), (len(k), len(z)))
        I2 = I1 if profile1 is profile2 else jnp.reshape(hm.mass_integral(k, z, profile2, bias_order=1), (len(k), len(z)))

        P_lin = hm.cosmology.pk(k, z, linear=True)
        # Ensure P_lin has shape (N_k, N_z)
        P_lin = jnp.reshape(P_lin, (len(k), -1))

        return jnp.squeeze(P_lin * I1 * I2)

    # ------------------------------------------------------------------
    # Combined 1-halo + 2-halo term
    # ------------------------------------------------------------------

    @jax.jit
    def pk_tot(self, halo_model, k, z, profile1, profile2=None):
        """
        Combine the 1-halo and 2-halo terms, with an HMcode-style smoothed
        transition between the two regimes.

        .. math::

            P(k, z) = \\left[ P_{1h}(k, z)^\\alpha + P_{2h}(k, z)^\\alpha \\right]^{1/\\alpha}

        where :math:`\\alpha` is :attr:`alpha_smooth`; :math:`\\alpha=1`
        recovers the plain sum :math:`P_{1h} + P_{2h}`. A term excluded via
        :attr:`include_1h`/:attr:`include_2h` is set to zero before combining,
        and since :math:`0^\\alpha = 0` for :math:`\\alpha > 0`, this reduces
        to the other term alone with no separate branch needed.

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
        pk_tot : array
            Combined power spectrum in :math:`\\mathrm{Mpc}^3`, with shape
            :math:`(N_k, N_z)`, where singleton dimensions get squeezed before
            return.
        """
        p1h = self.pk_1h(halo_model, k, z, profile1, profile2) if self.include_1h else 0.0
        p2h = self.pk_2h(halo_model, k, z, profile1, profile2) if self.include_2h else 0.0
        a = self.alpha_smooth

        return (p1h**a + p2h**a) ** (1.0 / a)


jax.tree_util.register_pytree_node(
    Pk,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: Pk._tree_unflatten(aux_data, children)
)
