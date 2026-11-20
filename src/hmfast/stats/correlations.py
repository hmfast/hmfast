"""
Real-space correlation functions: FFTLog transforms of a tabulated 3D power
spectrum or angular power spectrum, from any source (``Cosmology.pk``,
``Pk.pk_tot``, ``cl``, ``cl_linbias``, ...).
"""

import functools

import jax
import jax.numpy as jnp
import mcfit
import numpy as np

from hmfast.utils import cubic_interp_uniform


# Bessel order of the flat-sky Hankel kernel for each angular correlation type.
_ANGULAR_NU = {"NN": 0, "NG": 2, "GG+": 0, "GG-": 4}

def _log_grid(x, name):
    """Return ``x`` as a concrete 1D numpy array, checking it is log-uniformly spaced."""
    try:
        x = np.asarray(x, dtype=np.float64)
    except jax.errors.TracerArrayConversionError:
        raise ValueError(
            f"`{name}` must be concrete (e.g. a numpy array), not traced: the FFTLog plan is built from its values."
        ) from None
    if x.ndim != 1 or x.size < 2 or np.any(x <= 0):
        raise ValueError(f"`{name}` must be a 1D array of at least 2 positive values.")
    dlnx = np.diff(np.log(x))
    if not np.allclose(dlnx, dlnx[0], rtol=1e-6, atol=0.0):
        raise ValueError(f"`{name}` must be log-uniformly spaced, e.g. np.geomspace({x[0]:g}, {x[-1]:g}, {x.size}).")
    return x


@functools.lru_cache(maxsize=None)
def _transform(kind, x_bytes, nu, extrap):
    """Jitted mcfit transform along axis 0, built once per grid outside any trace."""
    x = np.frombuffer(x_bytes, dtype=np.float64)
    # mcfit plans on concrete values, so keep its setup concrete even when first reached under a trace.
    with jax.ensure_compile_time_eval():
        if kind == "P2xi":
            T = mcfit.P2xi(x, lowring=True, backend="jax")
        else:
            T = mcfit.C2w(x, nu=nu, lowring=True, backend="jax")
    return jax.jit(functools.partial(T, axis=0, extrap=extrap))


def _fftlog_interp(kind, x, f, y, nu, extrap):
    """Transform ``f`` (tabulated on ``x`` along axis 0) and interpolate the result onto ``y``."""
    f = jnp.asarray(f)
    if f.shape[0] != x.size:
        raise ValueError(f"Leading axis of the input ({f.shape[0]}) must match the grid length ({x.size}).")
    trailing = f.shape[1:]
    y_native, g_native = _transform(kind, x.tobytes(), nu, bool(extrap))(f.reshape(x.size, -1))

    # Linear in the output against ln y rather than log-log, so sign changes are safe
    ln_y, ln_y_native = jnp.log(jnp.atleast_1d(y)), jnp.log(y_native)
    g = jax.vmap(lambda col: jnp.interp(ln_y, ln_y_native, col), in_axes=1, out_axes=1)(g_native)
    return jnp.squeeze(g.reshape(ln_y.shape + trailing))


def _fullsky_kernel(type, x, lmax, binned):
    """
    (2l+1)/(4 pi) times the full-sky kernel of ``type`` for l = 0..lmax, shape (lmax+1, N), at x = cos(theta) or, if
    ``binned``, averaged over the bins between consecutive x, built from P_l and P_l' (Fang, Eifler & Krause 2020).
    """
    def step(carry, l):
        p0, p1, d0, d1 = carry  # P_{l-1}, P_l, P'_{l-1}, P'_l
        return (p1, ((2 * l + 1) * x * p1 - l * p0) / (l + 1), d1, d0 + (2 * l + 1) * p1), (p1, d1)

    _, (P, dP) = jax.lax.scan(step, (jnp.ones_like(x),) * 2 + (jnp.zeros_like(x),) * 2, jnp.arange(lmax + 2.0))
    l = jnp.arange(lmax + 1.0)[:, None]
    Pm, Pl, Pp = jnp.concatenate([P[:1], P[:-2]]), P[:-1], P[1:]
    dPm, dPl = jnp.concatenate([jnp.zeros_like(dP[:1]), dP[:-2]]), dP[:-1]
    sgn = {"GG+": 1.0, "GG-": -1.0}.get(type, 0.0)
    if binned:  # antiderivatives in x, differenced across each bin
        if type == "NN":
            F = (Pp - Pm) / (2 * l + 1)
        elif type == "NG":
            F = (l + 2 / (2 * l + 1)) * Pm + (2 - l) * x * Pl - 2 / (2 * l + 1) * Pp
        else:
            F = (-l * (l - 1) / 2 * ((l + 2 / (2 * l + 1)) * Pm + (2 - l) * x * Pl - 2 / (2 * l + 1) * Pp)
                 + (4 - l) * dPl + (l + 2) * (x * dPm - Pm) + sgn * (2 * (l - 1) * (x * dPl - Pl) - 2 * (l + 2) * dPm))
        K = (F[:, 1:] - F[:, :-1]) / (x[1:] - x[:-1])
    else:  # P_l, P_l^2 and G^+_{l,2} +- G^-_{l,2}, with P_l^2 = 2x P_l' - l(l+1) P_l
        P2l, P2m, s = 2 * x * dPl - l * (l + 1) * Pl, 2 * x * dPm - (l - 1) * l * Pm, 1 - x**2
        K = {"NN": Pl, "NG": P2l}.get(type, -((l - 4) / s + l * (l - 1) / 2) * P2l + (l + 2) * x / s * P2m
                                      + sgn * 2 / s * ((l - 1) * x * P2l - (l + 2) * P2m))
    f4 = jnp.where(l >= 2, (l - 1) * l * (l + 1) * (l + 2), 1.0)  # (l+2)!/(l-2)!
    norm = {"NN": 1.0, "NG": 1 / jnp.sqrt(f4)}.get(type, 2 / f4) * jnp.where((l >= 2) | (type == "NN"), 1.0, 0.0)
    return (2 * l + 1) / (4 * jnp.pi) * norm * K


def corr_3d(cosmology, k, pk, r, *, extrapolate=True):
    """
    3D correlation function of a tabulated power spectrum,

    .. math::

        \\xi(r) = \\frac{1}{2\\pi^2} \\int dk\\, k^2\\, P(k)\\, j_0(kr),

    by an FFTLog transform interpolated onto ``r``. Works for matter power
    spectra (:meth:`~hmfast.cosmology.Cosmology.pk`) and halo-model power
    spectra (:class:`~hmfast.stats.Pk`).

    Parameters
    ----------
    cosmology : Cosmology
        Cosmology object.
    k : array-like
        Wavenumbers in :math:`\\mathrm{Mpc}^{-1}`, shape :math:`(N_k,)`;
        concrete, positive and log-uniformly spaced.
    pk : jnp.ndarray
        Power spectrum in :math:`\\mathrm{Mpc}^3` on ``k``, shape :math:`(N_k,)` at
        one redshift or :math:`(N_k, N_z)` at :math:`N_z` redshifts.
    r : float or jnp.ndarray
        Comoving separations in :math:`\\mathrm{Mpc}`, scalar or shape :math:`(N_r,)`.
        Only reliable well inside :math:`[1/k_{\\max}, 1/k_{\\min}]`; FFTLog rings near the edges.
    extrapolate : bool
        If True (default), ``pk`` is extended as a power law beyond the ends of ``k`` before the transform;
        if False, it is padded with zeros.

    Returns
    -------
    xi : jnp.ndarray
        Correlation function with shape :math:`(N_r, N_z)`, where singleton
        dimensions get squeezed before return.
    """
    k = _log_grid(k, "k")
    return _fftlog_interp("P2xi", k, pk, r, 0, extrapolate)


def corr_angular(cosmology, l, cl, theta, *, type="NN", extrapolate=True, fullsky=False, bin_average=False):
    """
    Angular correlation function of a tabulated angular power spectrum, flat- or full-sky, at points or bin-averaged.

    Flat sky (default), by an FFTLog transform interpolated onto ``theta``:

    .. math::

        \\xi(\\theta) = \\int \\frac{d\\ell\\, \\ell}{2\\pi}\\, C_\\ell\\, J_\\nu(\\ell\\theta),

    with :math:`\\nu = 0, 2, 0, 4` for ``type`` = NN (e.g. :math:`w(\\theta)`), NG (:math:`\\gamma_t`),
    GG+ (:math:`\\xi_+`), GG- (:math:`\\xi_-`). Full sky (``fullsky=True``), exact at all angles:

    .. math::

        \\xi(\\theta) = \\sum_{\\ell=0}^{\\ell_{\\max}} \\frac{2\\ell+1}{4\\pi}\\, C_\\ell\\, d^\\ell_{m_1 m_2}(\\theta),

    with Wigner small-d orders :math:`(m_1, m_2) = (0,0), (2,0), (2,2), (2,-2)`, which tend to the
    :math:`J_\\nu` above at small angles; ``cl`` is interpolated (cubic, in :math:`\\ln\\ell`) onto every integer
    :math:`\\ell \\le \\ell_{\\max} = \\lfloor\\max\\ell\\rfloor` and taken to vanish outside the range of ``l``.
    Works for halo-model (:func:`~hmfast.stats.cl`) and linear-bias (:func:`~hmfast.stats.cl_linbias`) spectra.

    .. warning::

        The flat sky is sub-percent below a few degrees and increasingly wrong above: use ``fullsky=True``
        for large-angle clustering and galaxy-galaxy lensing. For clustering at large ``theta``, also
        compute :math:`C_\\ell` with ``l_limber > 0``, since Limber is inaccurate at low :math:`\\ell`.

    With ``bin_average=True`` (full sky only), ``theta`` holds bin edges and each output is :math:`\\xi` averaged
    over a bin weighted by solid angle (uniformly in :math:`\\cos\\theta`), exactly, from the closed-form
    bin-averaged Legendre functions of Fang, Eifler & Krause (2020).
    The full-sky projection matrix depends only on ``l`` and ``theta``,
    so under :func:`jax.jit` it is built once at trace time and folded into the compiled function;
    each call is then one interpolation and one matrix product.

    Shear :math:`C_\\ell` vanish at :math:`\\ell \\le 1`, so with ``extrapolate=True`` in the flat sky start
    ``l`` at :math:`\\ell \\ge 2`; power-law extrapolation of a zero gives NaN.

    Parameters
    ----------
    cosmology : Cosmology
        Cosmology object.
    l : array-like
        Multipoles, shape :math:`(N_\\ell,)`; concrete, positive and log-uniformly spaced.
    cl : jnp.ndarray
        Angular power spectrum on ``l``, shape :math:`(N_\\ell,)` for one spectrum
        or :math:`(N_\\ell, N_c)` for :math:`N_c` spectra stacked as columns
        (e.g. tomographic bin pairs).
    theta : float or jnp.ndarray
        Angular separations in degrees, scalar or shape :math:`(N_\\theta,)`; with ``bin_average``, increasing
        bin edges of shape :math:`(N_\\theta+1,)`. With ``fullsky``, concrete and within (0, 180) degrees.
        Flat sky is only reliable well inside :math:`[1/\\ell_{\\max}, 1/\\ell_{\\min}]` (radians);
        FFTLog rings near the edges.
    type : str, default "NN"
        Spin combination; ``type`` ∈ {"NN", "NG", "GG+", "GG-"} (static).
    extrapolate : bool
        Flat sky only. If True (default), ``cl`` is extended as a power law beyond the ends of ``l``
        before the transform; if False, it is padded with zeros.
    fullsky : bool, default False
        Whether to use the exact full-sky (Wigner-d) sum instead of the flat-sky transform (static).
    bin_average : bool, default False
        Whether ``theta`` holds bin edges to average over, instead of points; requires ``fullsky`` (static).

    Returns
    -------
    xi : jnp.ndarray
        Correlation function with shape :math:`(N_\\theta, N_c)`, where singleton
        dimensions get squeezed before return.
    """
    if type not in _ANGULAR_NU:
        raise ValueError(f"type must be one of {sorted(_ANGULAR_NU)}; got {type!r}.")
    l = _log_grid(l, "l")
    if not (fullsky or bin_average):
        return _fftlog_interp("C2w", l, cl, jnp.deg2rad(theta), _ANGULAR_NU[type], extrapolate)

    if bin_average and not fullsky:
        raise ValueError("bin_average requires fullsky=True; average flat-sky points directly instead.")
    try:
        th = np.deg2rad(np.atleast_1d(np.asarray(theta, dtype=np.float64)))
    except jax.errors.TracerArrayConversionError:
        raise ValueError("`theta` must be concrete with fullsky=True.") from None
    if np.any(th <= 0) or np.any(th >= np.pi) or (bin_average and (th.size < 2 or np.any(np.diff(th) <= 0))):
        raise ValueError("`theta` must lie in (0, 180) degrees, and be increasing edges with bin_average.")
    if l.size < 4:
        raise ValueError("`l` needs at least 4 multipoles with fullsky=True.")

    trailing, cl = jnp.shape(cl)[1:], jnp.asarray(cl).reshape(l.size, -1)
    ell = np.arange(int(l[-1]) + 1, dtype=np.float64)
    with jax.ensure_compile_time_eval():  # theta and l are concrete: build the matrix eagerly, not in the trace
        M = _fullsky_kernel(type, jnp.cos(jnp.asarray(th)), ell.size - 1, bin_average)
    ln_l = np.log(l)
    cl_int = cubic_interp_uniform(np.log(np.clip(ell, l[0], l[-1]))[:, None], ln_l[0], (ln_l[-1] - ln_l[0]) / (l.size - 1),
                                  cl, cl[0], cl[-1])
    cl_int = jnp.where((ell >= l[0])[:, None], cl_int, 0.0)  # C_l vanishes below the grid
    return jnp.squeeze((M.T @ cl_int).reshape(M.shape[1:] + trailing))
