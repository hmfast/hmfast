"""
Non-Limber angular power spectrum for the 2-halo term.

Implements a SwiftCl-style (Reymond, Reeves, Zhang, Refregier, arXiv:2505.22718)
FFTLog decomposition of the tracer kernel, exploiting the separable structure
already present in hmfast's own 2-halo model,

    P_2h(k, z) = P_lin(k, z) * I_1^1(k, z)^2,

to compute the "beyond-Limber" unequal-time projection without the double
line-of-sight integral against oscillatory spherical Bessel functions. The
1-halo term is not treated here -- it has no clean unequal-time
generalisation and dominates only at high ell, where the existing Limber
``Pk.cl_1h`` is already accurate.

Notes
-----
- Differentiable via ``jax.grad``/``jax.jacobian`` (verified against finite
  differences), but not ``jax.jit``-able: the closed-form Hankel-transform
  Gamma-ratio table is built with ``scipy.special.loggamma`` in plain numpy,
  mirroring the ``mcfit.kernels`` convention already used by
  ``Pk.xi_1h``/``xi_2h``, which requires ``l`` and everything feeding
  ``chi_min``/``chi_max`` (hence any cosmology parameter, via
  ``angular_diameter_distance``) to be concrete rather than traced.
- Evaluating tracer kernels or ``P(k, z)`` beyond the trained emulator's
  background/z grids (e.g. CMB lensing out to ``z_star``) requires the
  ``halo_model.cosmology`` passed in to already have ``extrapolate_z=True``
  set by the caller -- exactly the same convention ``Pk.cl_1h``/``cl_2h``
  already rely on.
- Each kernel is windowed with a smooth taper before FFTLog decomposition
  (see :func:`_tail_taper`) to suppress FFT periodicity ringing, and the
  final k-quadrature masks out (k, ell) pairs whose Bessel turning point
  (ell+0.5)/k falls outside [chi_min, chi_max] -- the closed-form Hankel
  transform of each power-law term is only trustworthy there; outside it,
  it is dominated by numerical cancellation artifacts rather than signal.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import loggamma

from hmfast.stats import Pk
from hmfast.tracers.cmb_lensing import CMBLensingTracer

jax.config.update("jax_enable_x64", True)


# FFTLog / Hankel-transform primitives

def _fftlog_log_grid(chi_min, chi_max, n_fft):
    """Fixed, static log-spaced chi nodes spanning [chi_min, chi_max]."""
    return jnp.geomspace(chi_min, chi_max, n_fft)


def _tail_taper(chi, chi_min, chi_max, frac=0.1):
    """Raised-cosine window: 1 in the interior, smoothly to 0 at chi_min/chi_max over a fraction ``frac`` of the log-chi span."""
    ln_chi = jnp.log(chi)
    ln_min, ln_max = jnp.log(chi_min), jnp.log(chi_max)
    edge = frac * (ln_max - ln_min)

    left = 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip((ln_chi - ln_min) / edge, 0.0, 1.0)))
    right = 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip((ln_max - ln_chi) / edge, 0.0, 1.0)))
    return jnp.minimum(left, right)


def _fftlog_biased_coeffs(f_chi, chi_min, chi_max, n_fft, bias):
    """FFTLog-decompose ``f_chi`` (on :func:`_fftlog_log_grid`) into c_n such that f_chi(chi) ~= sum_n c_n * chi**(bias + 1j*eta_n)."""
    chi = _fftlog_log_grid(chi_min, chi_max, n_fft)
    d_ln_chi = jnp.log(chi_max / chi_min) / (n_fft - 1)
    eta_n = 2.0 * jnp.pi * jnp.fft.fftfreq(n_fft, d=d_ln_chi)

    g = f_chi * chi ** (-bias)
    c_n = jnp.fft.fft(g, axis=-1) / n_fft
    c_n = c_n * chi_min ** (-1j * eta_n)
    return c_n, eta_n


def _gamma_ratio_table(l, bias, eta_n):
    """Closed-form Hankel-transform coefficients A_n(l) = 2**(p_n-1)*sqrt(pi)*Gamma((1+l+p_n)/2)/Gamma((2+l-p_n)/2), p_n = bias + 1j*eta_n."""
    l = np.atleast_1d(np.asarray(l, dtype=np.float64))
    eta_n = np.asarray(eta_n, dtype=np.float64)
    p = bias + 1j * eta_n  # (n_fft,)

    log_num = loggamma(0.5 * (1.0 + l[:, None] + p[None, :]))
    log_den = loggamma(0.5 * (2.0 + l[:, None] - p[None, :]))
    return (2.0 ** (p[None, :] - 1.0)) * np.sqrt(np.pi) * np.exp(log_num - log_den)


# Comoving-distance inversion and the D(k, z) growth/bias factor

def _invert_chi_to_z(cosmology, chi_targets, z_max=1200.0, n_tail=500):
    """Invert chi(z) = angular_diameter_distance(z)*(1+z) for chi_targets via interpolation on a dense z grid out to z_max."""
    z_bg = cosmology._z_grid_bg()
    z_tail = jnp.geomspace(z_bg[-1] + 1.0, z_max, n_tail)
    z_grid = jnp.concatenate([z_bg, z_tail])
    chi_grid = cosmology.angular_diameter_distance(z_grid) * (1.0 + z_grid)
    return jnp.interp(chi_targets, chi_grid, z_grid)


def _D_kz(halo_model, profile, k, z, z_fid=0.0):
    """D(k,z) = sqrt(P_lin(k,z)/P_lin(k,z_fid)) * I_1^1(k,z), the per-time factor of the separable P_2h(k;z1,z2) ~= P_lin(k,z_fid)*D(k,z1)*D(k,z2) ansatz."""
    cosmology = halo_model.cosmology
    k, z = jnp.atleast_1d(k), jnp.atleast_1d(z)

    z_b = cosmology._z_grid_pk()[-1]
    in_bounds = z <= z_b
    growth_ratio = jnp.where(in_bounds, 1.0, cosmology.growth_factor(z) / cosmology.growth_factor(z_b))
    z_eval = jnp.where(in_bounds, z, z_b)

    I1 = jnp.reshape(halo_model._I(profile, k, z_eval, bias_order=1), (len(k), len(z)))

    Plin_zeval = jnp.reshape(cosmology.pk(k, z_eval, linear=True), (len(k), len(z)))
    Plin_zfid = jnp.reshape(cosmology.pk(k, jnp.atleast_1d(z_fid), linear=True), (len(k), 1))

    return growth_ratio[None, :] * jnp.sqrt(Plin_zeval / Plin_zfid) * I1


# Non-Limber 2-halo term (private; use the public cl_2h below)

def _cl_2h_nonlimber(
    halo_model, tracer1, tracer2, l, z,
    k=None,
    z_fid=0.0,
    n_chi=None, n_interp=200,
    bias=0.1,
    taper_frac=0.1,
):
    """
    Non-Limber 2-halo C_ell via an FFTLog/closed-form Hankel transform.

    ``l`` must be concrete (non-traced): part of the calculation runs in
    plain numpy. ``z`` only sets the range/resolution of an internal chi
    grid (via min, max, length) and is auto-widened to cover each tracer's
    own support. ``k`` defaults to the cosmology's native P(k) grid.
    ``n_chi``/``n_interp`` control internal grid resolutions; ``bias``
    (<1) is the FFTLog de-trending exponent; ``taper_frac`` is the
    fraction of the chi grid tapered to zero at each edge before the FFT.
    ``tracer1``/``tracer2`` share one chi grid, so they can't use
    independent z-ranges.
    """
    tracer2 = tracer1 if tracer2 is None else tracer2
    cosmology = halo_model.cosmology

    l_arr = np.atleast_1d(np.asarray(l, dtype=np.float64))
    z = jnp.atleast_1d(z)
    z_min = float(jnp.min(z))
    z_max = float(jnp.max(z))
    n_chi = n_chi if n_chi is not None else len(z)

    # Widen z_max (never narrow) to cover each tracer's own declared support.
    z_max = max([z_max, *(float(t.z_max) for t in (tracer1, tracer2) if hasattr(t, "z_max")),
                 *(float(jnp.max(t.dndz[0])) for t in (tracer1, tracer2) if hasattr(t, "dndz"))])

    involves_cmb_lensing = isinstance(tracer1, CMBLensingTracer) or isinstance(tracer2, CMBLensingTracer)
    chi_star = None
    if involves_cmb_lensing:
        derived = cosmology.derived_parameters()
        chi_star = float(derived["chi_star"])
        z_max = max(z_max, 1.05 * float(derived["z_star"]))  # margin gives the taper room past chi_star

    # Non-differentiable hyperparameters (matching `l`'s own convention), needed concrete for the numpy Gamma table.
    chi_min = jax.lax.stop_gradient(cosmology.angular_diameter_distance(z_min) * (1.0 + z_min))
    chi_max = jax.lax.stop_gradient(cosmology.angular_diameter_distance(z_max) * (1.0 + z_max))

    if k is None:
        k, _ = cosmology._pk_grid()
    k_fine = jnp.asarray(k)
    k_min, k_max = k_fine[0], k_fine[-1]

    chi_nodes = _fftlog_log_grid(chi_min, chi_max, n_chi)
    z_nodes = jnp.minimum(_invert_chi_to_z(cosmology, chi_nodes), z_max)  # guard chi->z round-trip overshoot past z_max

    k_anchors = jnp.geomspace(k_min, k_max, n_interp)
    log_ka, log_kf = jnp.log(k_anchors), jnp.log(k_fine)

    def kernel_on_chi_grid(tracer):
        W = jnp.atleast_1d(tracer.kernel(cosmology, z_nodes))
        if isinstance(tracer, CMBLensingTracer):
            W = jnp.where(chi_nodes < chi_star, W, 0.0)
        return W

    taper = _tail_taper(chi_nodes, chi_min, chi_max, frac=taper_frac)

    def fftlog_coeffs_fine(W, profile):
        D_anchor = _D_kz(halo_model, profile, k_anchors, z_nodes, z_fid=z_fid)  # (n_interp, n_chi)
        f_chi = W[None, :] * D_anchor * taper[None, :]
        c_n, eta_n = _fftlog_biased_coeffs(f_chi, chi_min, chi_max, n_chi, bias)  # (n_interp, n_chi)

        interp_col = lambda col: jnp.interp(log_kf, log_ka, col)
        c_n_fine = (
            jax.vmap(interp_col, in_axes=1, out_axes=1)(jnp.real(c_n))
            + 1j * jax.vmap(interp_col, in_axes=1, out_axes=1)(jnp.imag(c_n))
        )
        return c_n_fine, eta_n

    W1 = kernel_on_chi_grid(tracer1)
    c1, eta_n = fftlog_coeffs_fine(W1, tracer1.profile)

    if tracer2 is tracer1:
        c2 = c1
    else:
        W2 = kernel_on_chi_grid(tracer2)
        c2, _ = fftlog_coeffs_fine(W2, tracer2.profile)

    A_table = jnp.asarray(_gamma_ratio_table(l_arr, bias, np.asarray(jax.lax.stop_gradient(eta_n))))  # (N_ell, n_chi)

    def delta_ell(c_n_fine):
        kp = k_fine[:, None] ** (-1.0 - bias) * jnp.exp(-1j * eta_n[None, :] * log_kf[:, None])
        terms = c_n_fine * kp  # (n_k, n_chi)
        return jnp.real(jnp.einsum('en,kn->ke', A_table, terms))  # (n_k, N_ell)

    Delta1 = delta_ell(c1)
    Delta2 = Delta1 if c2 is c1 else delta_ell(c2)

    # Delta_l(k) is only trustworthy where its Bessel turning point (l+0.5)/k falls within [chi_min, chi_max].
    resonant_chi = (jnp.asarray(l_arr)[None, :] + 0.5) / k_fine[:, None]  # (n_k, N_ell)
    validity_mask = (resonant_chi >= chi_min) & (resonant_chi <= chi_max)

    Plin_fid = jnp.reshape(cosmology.pk(k_fine, jnp.atleast_1d(z_fid), linear=True), (k_fine.shape[0],))

    integrand = k_fine[:, None] ** 2 * Plin_fid[:, None] * Delta1 * Delta2 * validity_mask
    Cl = (2.0 / jnp.pi) * jnp.trapezoid(integrand * k_fine[:, None], x=log_kf, axis=0)

    return jnp.squeeze(Cl)


def _cl_limber(halo_model, tracer1, tracer2, l, z, include_1h=False, include_2h=True, k_damp=0.01, pk=None):
    """
    Limber C_ell for either or both halo terms; reproduces Pk.cl_2h with
    the defaults, or Pk.cl_1h with include_1h=True, include_2h=False.
    Unlike ``_cl_2h_nonlimber``, ``l`` may be traced. jit-able only with a
    pre-built ``pk`` (``Pk()`` itself is not jit-traceable).
    """
    cosmology = halo_model.cosmology
    tracer2 = tracer1 if tracer2 is None else tracer2
    pk = pk if pk is not None else Pk()
    z = jnp.atleast_1d(z)
    z_b = cosmology._z_grid_pk()[-1]

    # growth_factor is evaluated on the full z array, not per-scalar inside vmap, to match Pk.cl_1h/cl_2h exactly.
    in_bounds = z <= z_b
    growth_ratio_sq = jnp.where(in_bounds, 1.0, (cosmology.growth_factor(z) / cosmology.growth_factor(z_b)) ** 2)
    z_eval = jnp.where(in_bounds, z, z_b)

    def get_pk_slice(zi, zi_eval):
        chi_i = cosmology.angular_diameter_distance(zi) * (1.0 + zi)
        ki = (l + 0.5) / chi_i
        zi_eval = jnp.atleast_1d(zi_eval)

        p = 0.0
        if include_1h:
            p = p + pk.pk_1h(halo_model, ki, zi_eval, tracer1.profile, tracer2.profile, k_damp=k_damp)
        if include_2h:
            p = p + pk.pk_2h(halo_model, ki, zi_eval, tracer1.profile, tracer2.profile)
        return jnp.atleast_1d(p).flatten()

    P_grid = jax.vmap(get_pk_slice)(z, z_eval) * growth_ratio_sq[:, None]
    kernel1 = jnp.atleast_1d(tracer1.kernel(cosmology, z))
    kernel2 = jnp.atleast_1d(tracer2.kernel(cosmology, z))
    chi = cosmology.angular_diameter_distance(z) * (1.0 + z)
    limber_weight = cosmology.comoving_volume_element(z) / chi**4

    integrand = P_grid * (limber_weight[:, None] * kernel1[:, None] * kernel2[:, None])
    return jnp.squeeze(jnp.trapezoid(integrand, x=z, axis=0))


def cl_2h(halo_model, tracer1, tracer2, l, z, l_limber=0.0, pk=None,
          k=None, z_fid=0.0, n_chi=None, n_interp=200, bias=0.1, taper_frac=0.1):
    """
    2-halo angular power spectrum C_ell, exact at low ell and via the fast
    Limber approximation at high ell.

    The Limber approximation assumes each tracer's kernel varies slowly
    compared to the density fluctuations it's weighted against; this
    breaks down at low multipoles, where kernels can be narrow or only
    partially overlap in redshift. Below `l_limber`, this function instead
    performs an exact projection via a closed-form Hankel-transform
    (FFTLog) decomposition of each kernel; at and above `l_limber`, it
    falls back to the standard, much cheaper Limber integral. Only the
    2-halo contribution is computed here; add a 1-halo term separately
    (e.g. from ``Pk.cl_1h``) for the full spectrum.

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
        JAX), since choosing which method to use per multipole happens
        outside JAX's tracing.
    z : array-like
        Redshift sampling. Only its minimum, maximum, and length are used,
        to set up internal grids; each tracer's own redshift support is
        covered automatically even if narrower than `z`, so it's safe to
        reuse whatever `z` grid you'd use elsewhere for these tracers.
    l_limber : float, default 0.0
        Multipoles below this use the exact low-ell method; multipoles at
        or above it use the Limber approximation. The default, 0.0, uses
        Limber everywhere; raise it to switch to the exact method at low
        ell, where it matters most.
    pk : Pk or None, default None
        Optional pre-built ``Pk`` instance used for the Limber part of the
        calculation; reuse one across calls to avoid rebuilding its
        internal setup. A new instance is created automatically if omitted.
    k : array-like or None, default None
        Wavenumber grid used internally by the exact low-ell calculation.
        Defaults to the cosmology's own tabulated power-spectrum grid.
    z_fid : float, default 0.0
        Fiducial redshift used to normalize the exact low-ell calculation's
        internal model of how the power spectrum evolves with redshift.
    n_chi : int or None, default None
        Resolution of the internal comoving-distance grid used by the exact
        low-ell calculation. Defaults to ``len(z)``.
    n_interp : int, default 200
        Number of coarse wavenumber points at which the exact low-ell
        calculation evaluates its most expensive step, before interpolating
        onto the full resolution set by `k`.
    bias : float, default 0.1
        Technical exponent (must be less than 1) controlling how the exact
        low-ell calculation numerically decomposes each kernel; the default
        works well across tracer types and rarely needs changing.
    taper_frac : float, default 0.1
        Fraction of the internal comoving-distance grid, at each end, over
        which the exact low-ell calculation smoothly tapers the kernel to
        zero to avoid edge artifacts.

    Returns
    -------
    cl_2h : array, shape (N_ell,)
        The 2-halo angular power spectrum, in the same order as `l`.

    Notes
    -----
    `l` must be concrete because choosing the method per multipole happens
    outside JAX's tracing, so this function cannot be wrapped in
    ``jax.jit`` as a whole. Gradients via ``jax.grad``/``jax.jacobian``
    with respect to any other argument work normally.
    """
    l_np = np.atleast_1d(np.asarray(l, dtype=np.float64))
    idx_low, idx_high = np.where(l_np < l_limber)[0], np.where(l_np >= l_limber)[0]

    result = jnp.zeros(l_np.shape[0])
    if idx_low.size:
        result = result.at[idx_low].set(jnp.atleast_1d(
            _cl_2h_nonlimber(halo_model, tracer1, tracer2, l_np[idx_low], z,
                             k=k, z_fid=z_fid, n_chi=n_chi, n_interp=n_interp, bias=bias, taper_frac=taper_frac)))
    if idx_high.size:
        result = result.at[idx_high].set(jnp.atleast_1d(
            _cl_limber(halo_model, tracer1, tracer2, l_np[idx_high], z, include_2h=True, pk=pk)))
    return result
