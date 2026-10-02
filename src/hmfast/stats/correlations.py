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


def corr_3d(k, pk, r, extrap=True):
    """
    3D correlation function of a tabulated power spectrum,

    .. math::

        \\xi(r) = \\frac{1}{2\\pi^2} \\int dk\\, k^2\\, P(k)\\, j_0(kr),

    by an FFTLog transform (:class:`mcfit.P2xi`) interpolated onto ``r``.
    Works for any :math:`P(k)`, e.g. :meth:`Cosmology.pk` or
    :meth:`~hmfast.stats.pk.Pk.pk_tot`. Not jitted itself, but traceable
    inside a user's ``jax.jit``/``jax.grad`` as long as ``k`` stays concrete.

    Parameters
    ----------
    k : array-like
        Wavenumber grid in :math:`\\mathrm{Mpc}^{-1}`; concrete and
        log-uniform, e.g. ``np.geomspace(1e-4, 50, 1000)``.
    pk : jnp.ndarray
        Power spectrum in :math:`\\mathrm{Mpc}^3`, shape :math:`(N_k, ...)`;
        trailing axes (e.g. redshift) are transformed independently.
    r : float or jnp.ndarray
        Comoving separation in :math:`\\mathrm{Mpc}`. Only reliable well
        inside :math:`[1/k_{\\max}, 1/k_{\\min}]`; FFTLog rings near the edges.
    extrap : bool, default True
        Power-law extrapolate ``pk`` beyond the ends of ``k`` before transforming.

    Returns
    -------
    xi : jnp.ndarray
        Correlation function with shape :math:`(N_r, ...)`, where singleton
        dimensions get squeezed before return.
    """
    k = _log_grid(k, "k")
    return _fftlog_interp("P2xi", k, pk, r, 0, extrap)


def corr_angular(l, cl, theta, type="NN", extrap=True):
    """
    Flat-sky angular correlation function of a tabulated angular power spectrum,

    .. math::

        \\xi(\\theta) = \\int \\frac{d\\ell\\, \\ell}{2\\pi}\\, C_\\ell\\, J_\\nu(\\ell\\theta),

    by an FFTLog transform (:class:`mcfit.C2w`) interpolated onto ``theta``,
    with :math:`\\nu = 0, 2, 0, 4` for ``type`` = NN (e.g. :math:`w(\\theta)`),
    NG (:math:`\\gamma_t`), GG+ (:math:`\\xi_+`), GG- (:math:`\\xi_-`).
    Works for any :math:`C_\\ell`, e.g. :func:`~hmfast.stats.cl` or
    :func:`~hmfast.stats.cl_linbias`. Not jitted itself, but traceable
    inside a user's ``jax.jit``/``jax.grad`` as long as ``l`` stays concrete.

    .. warning::

        This is the flat-sky (small-angle) approximation: sub-percent below a
        few degrees, increasingly wrong above. Do not use it for large-angle
        measurements without checking against a full-sky Legendre sum.

    For clustering at large ``theta``, compute :math:`C_\\ell` with
    ``l_limber > 0``, since Limber is inaccurate at low :math:`\\ell`. Shear
    :math:`C_\\ell` vanish at :math:`\\ell \\le 1`, so with ``extrap=True`` start
    ``l`` at :math:`\\ell \\ge 2`; power-law extrapolation of a zero gives NaN.

    Parameters
    ----------
    l : array-like
        Multipole grid; concrete and log-uniform, e.g. ``np.geomspace(1, 1e5, 512)``.
    cl : jnp.ndarray
        Angular power spectrum, shape :math:`(N_\\ell, ...)`; trailing axes
        (e.g. tomographic pairs) are transformed independently.
    theta : float or jnp.ndarray
        Angular separation in degrees. Only reliable well inside
        :math:`[1/\\ell_{\\max}, 1/\\ell_{\\min}]` (radians); FFTLog rings near the edges.
    type : str, default "NN"
        Spin combination; ``type`` ∈ {"NN", "NG", "GG+", "GG-"} (static).
    extrap : bool, default True
        Power-law extrapolate ``cl`` beyond the ends of ``l`` before transforming.

    Returns
    -------
    xi : jnp.ndarray
        Correlation function with shape :math:`(N_\\theta, ...)`, where
        singleton dimensions get squeezed before return.
    """
    if type not in _ANGULAR_NU:
        raise ValueError(f"type must be one of {sorted(_ANGULAR_NU)}; got {type!r}.")
    l = _log_grid(l, "l")
    return _fftlog_interp("C2w", l, cl, jnp.deg2rad(theta), _ANGULAR_NU[type], extrap)
