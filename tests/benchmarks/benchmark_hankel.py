"""
Numerical accuracy of the generic Hankel-transform profile path (GNFW, B12, B16), against independent truths: the
analytic truncated-NFW transform and quad integrals of the truncated GNFW. Moved out of the unit suite because the
quad references and n_x up to 400 make it slow.
"""

import functools

import jax
import jax.numpy as jnp
import mcfit
import numpy as np
import pytest
from scipy.integrate import quad

from hmfast.halos import HaloModel
from hmfast.halos.concentration import D08Concentration
from hmfast.halos.massdef import MassDefinition
from hmfast.halos.profiles import GNFWPressureProfile, NFWMatterProfile
from hmfast.halos.profiles.base_profile import HaloProfile, HankelTransform

M_GRID = jnp.geomspace(1e10, 1e15, 40)


@pytest.fixture
def hm200c(fixed_cosmology):
    """A HaloModel at 200c/D08, the native mass definition of the NFW accuracy checks."""
    return HaloModel(
        cosmology=fixed_cosmology,
        mass_def=MassDefinition(200, "critical"),
        concentration=D08Concentration(),
        m_range=(M_GRID[0], M_GRID[-1]),
        n_m=M_GRID.shape[0],
    )


@pytest.fixture
def hm500c(fixed_cosmology):
    """A HaloModel at 500c/D08, the native mass definition of the GNFW accuracy checks."""
    return HaloModel(cosmology=fixed_cosmology, mass_def=MassDefinition(500, "critical"), concentration=D08Concentration())


# ---------------------------------------------------------------------------------------------------------------------
# Numerical accuracy of the generic Hankel path, HaloProfile._fourier_via_hankel_transform (GNFW, B12, B16).
# Every test compares against an independent truth -- the analytic truncated-NFW transform, or a quad integral of the
# truncated GNFW -- never another halo-model code. Thresholds are ~2x the error measured when written (noted per test),
# so a change that degrades the transform by ~2x fails; lower them when accuracy improves. Errors are the maximum
# absolute error over the targets divided by u(q -> 0).
# ---------------------------------------------------------------------------------------------------------------------

# Off-grid targets in the dimensionless wavenumber q = k r_scale (1+z).
Q = np.geomspace(1e-2, 20.0, 97)
Q_LOW = np.geomspace(1e-4, 1e-2, 9)
GNFW_SHAPE = dict(c500=1.81, alpha=1.062, beta=4.13, gamma=0.3292)
X_MIN, X_MAX = 1e-5, 4.0


def _gnfw_shape(x):
    c, a, b, g = GNFW_SHAPE["c500"], GNFW_SHAPE["alpha"], GNFW_SHAPE["beta"], GNFW_SHAPE["gamma"]
    return (c * x) ** -g * (1.0 + (c * x) ** a) ** ((g - b) / a)


def _gnfw_truth_at(q, x_out):
    """int_0^x_out x^2 f(x) j0(q x) dx, to ~1e-12."""
    if q < 1e-8:
        return quad(lambda x: x * x * _gnfw_shape(x), 1e-14, x_out, limit=500, epsabs=0, epsrel=1e-13)[0]
    return quad(lambda x: x * _gnfw_shape(x) / q, 1e-14, x_out, weight="sin", wvar=q,
                limit=2000, epsabs=0, epsrel=1e-13)[0]


@functools.lru_cache(maxsize=None)
def _gnfw_truth(x_out, which="Q"):
    qs = Q if which == "Q" else Q_LOW
    return _gnfw_truth_at(0.0, x_out), np.array([_gnfw_truth_at(q, x_out) for q in qs])


def _gnfw(n_x, x_out, B=1.0):
    return GNFWPressureProfile(x_range=(X_MIN, X_MAX), n_x=n_x, B=B, x_out=x_out, **GNFW_SHAPE)


def _gnfw_u(hm, n_x, x_out, q, m=1e14, z=0.0):
    """hmfast's GNFW transform at q, divided by its constant prefactor so it compares to the truth."""
    p = _gnfw(n_x, x_out)
    m1, z1 = jnp.atleast_1d(m), jnp.atleast_1d(z)
    rs = float(jnp.squeeze(p._fourier_radius_scale(hm, m1, z1)))
    amp = float(jnp.squeeze(p.real(hm, jnp.array([0.5 * rs * (1 + z)]), m, z))) / _gnfw_shape(0.5)
    u = np.asarray(p.fourier(hm, jnp.asarray(q / (rs * (1 + z))), m, z))
    return u / (4 * np.pi * (rs * (1 + z)) ** 3 * amp)


def gnfw_error(hm, n_x, x_out, which="Q"):
    u0, truth = _gnfw_truth(x_out, which)
    u = _gnfw_u(hm, n_x, x_out, Q if which == "Q" else Q_LOW)
    return np.max(np.abs(u - truth)) / u0


def mcfit_ideal_error(n_x, pad_decades=2.0):
    """Best achievable with mcfit on the same padded grid: x_out on the last node at half weight, read at mcfit's own nodes."""
    x = np.logspace(np.log10(X_MIN), np.log10(X_MAX), n_x)
    h = np.log(x[1] / x[0])
    n_pad = int(np.ceil(pad_decades * np.log(10) / h))
    F = _gnfw_shape(x) * x**0.5
    F[-1] *= 0.5
    q, G = mcfit.Hankel(x[0] * np.exp(h * np.arange(n_x + n_pad)), nu=0.5, lowring=True, backend="jax")(
        jnp.asarray(np.concatenate([F, np.zeros(n_pad)])), extrap=False)
    q, G = np.asarray(q), np.asarray(G)
    sel = (q > Q[0]) & (q < Q[-1])
    u = (G * np.sqrt(np.pi / (2 * q)))[sel]
    truth = np.array([_gnfw_truth_at(v, X_MAX) for v in q[sel]])
    return np.max(np.abs(u - truth)) / _gnfw_truth(X_MAX)[0]


def nfw_error(hm, n_x, x_max, m, z):
    """Truncated NFW pushed through the generic Hankel path, against its analytic transform."""
    nfw = NFWMatterProfile()
    x = jnp.logspace(np.log10(X_MIN), np.log10(x_max), n_x)
    nfw._hankel, nfw.x_grid, nfw.x_out = HankelTransform(x, nu=0.5), x, 1.0  # x = r / r_delta, truncated at r_delta
    m1, z1 = jnp.atleast_1d(m), jnp.atleast_1d(z)
    r_delta = jnp.reshape(hm.mass_def.r_delta(hm.cosmology, m1, z1), (1, 1))
    k = jnp.asarray(Q) / (r_delta[0, 0] * (1 + z))
    u_hankel = np.asarray(HaloProfile._fourier_via_hankel_transform(nfw, hm, k, m1, z1, r_delta))
    u_exact = np.asarray(nfw.fourier(hm, k, m, z))
    return np.max(np.abs(u_hankel - u_exact)) / np.max(np.abs(u_exact))


NFW_HALOS = [(1e12, 0.0), (1e14, 0.5), (1e15, 1.0)]


class TestHankelAgainstAnalyticNFW:
    # x_out = 1 on the last grid node (measured max over halos: 1.5e-3 at n_x=100).
    @pytest.mark.parametrize("m,z", NFW_HALOS)
    def test_truncation_on_grid_edge(self, hm200c, m, z):
        assert nfw_error(hm200c, 100, 1.0, m, z) < 3e-3

    # x_out = 1 between nodes of a grid running to 1.5 (measured max over halos: 3.2e-3 at n_x=100).
    @pytest.mark.parametrize("m,z", NFW_HALOS)
    def test_truncation_between_nodes(self, hm200c, m, z):
        assert nfw_error(hm200c, 100, 1.5, m, z) < 6e-3

    # Second-order convergence: doubling n_x cuts the error by >= 3 (measured 4.7).
    def test_convergence_rate(self, hm200c):
        e = [nfw_error(hm200c, n, 1.0, 1e14, 0.5) for n in (100, 200, 400)]
        assert e[0] / e[1] > 3 and e[1] / e[2] > 3


class TestHankelAgainstQuadGNFW:
    # x_out on the grid edge, the GNFW default (measured 5.2e-4 / 1.3e-4 at n_x=100 / 200).
    @pytest.mark.parametrize("n_x,tol", [(100, 1e-3), (200, 2.6e-4)])
    def test_truncation_on_grid_edge(self, hm500c, n_x, tol):
        assert gnfw_error(hm500c, n_x, X_MAX) < tol

    # x_out between nodes (measured 1.7e-3 / 1.3e-4 at n_x=100 / 200).
    @pytest.mark.parametrize("n_x,tol", [(100, 3.4e-3), (200, 2.7e-4)])
    def test_truncation_between_nodes(self, hm500c, n_x, tol):
        assert gnfw_error(hm500c, n_x, 3.0) < tol

    # x_out just inside the last node, where it is not snapped onto it (measured 2.4e-3 at n_x=100).
    def test_truncation_just_inside_node(self, hm500c):
        assert gnfw_error(hm500c, 100, X_MAX * (1 - 1e-3)) < 5e-3

    # Second-order convergence: doubling n_x cuts the error by >= 3 (measured 4.1).
    def test_convergence_rate(self, hm500c):
        e = [gnfw_error(hm500c, n, X_MAX) for n in (100, 200, 400)]
        assert e[0] / e[1] > 3 and e[1] / e[2] > 3

    # Below mcfit's native q range, where the zero padding and the q -> 0 anchor take over (measured 3.8e-4).
    def test_low_q(self, hm500c):
        assert gnfw_error(hm500c, 100, X_MAX, which="low") < 8e-4


class TestHankelMatchesMcfit:
    # hmfast adds no significant error on top of mcfit's own (measured ratio 1.09-1.11).
    @pytest.mark.parametrize("n_x", [100, 200, 400])
    def test_no_loss_relative_to_mcfit(self, hm500c, n_x):
        assert gnfw_error(hm500c, n_x, X_MAX) / mcfit_ideal_error(n_x) < 1.25


class TestHankelEdgeCases:
    # x_out = inf truncates at the grid edge, so it must equal x_out = x_max (round-off in real() must not drop the node).
    def test_untruncated_equals_truncated_at_grid_edge(self, hm500c):
        u_inf = _gnfw_u(hm500c, 100, jnp.inf, Q)
        u_edge = _gnfw_u(hm500c, 100, X_MAX, Q)
        assert np.allclose(u_inf, u_edge, rtol=1e-9, atol=0)

    # x_out a few ulps below the last node snaps onto it; real() must still see that node as inside (1.8e-2 if it doesn't).
    def test_node_kept_when_x_out_rounds_below_it(self, hm500c):
        u_below = _gnfw_u(hm500c, 100, X_MAX * (1 - 1e-14), Q)
        u_edge = _gnfw_u(hm500c, 100, X_MAX, Q)
        assert np.allclose(u_below, u_edge, rtol=1e-9, atol=0)

    # Far beyond the native grid the transform clamps to its last value rather than extrapolating.
    def test_clamps_above_native_grid(self, hm500c):
        u = _gnfw_u(hm500c, 100, X_MAX, np.array([1e7, 1e9]))
        assert np.all(np.isfinite(u)) and u[0] == u[1]


class TestHankelSmoothness:
    # d ln u / d ln B varies smoothly along a dense sweep in B, so the interpolation onto q = k r_500 has no kinks
    # (measured max second difference 1.3e-4; linear interpolation gives 2.2e-2).
    def test_autodiff_derivative_has_no_kinks_in_B(self, hm500c):
        k = jnp.array([0.3, 1.0, 3.0])
        p = _gnfw(100, X_MAX)
        dlnu_dlnB = jax.vmap(jax.jacfwd(lambda lnB: jnp.log(p.update(B=jnp.exp(lnB)).fourier(hm500c, k, 1e14, 0.0))))
        d = np.asarray(dlnu_dlnB(jnp.linspace(0.0, np.log(1.5), 201)))
        assert np.max(np.abs(d[2:] - 2 * d[1:-1] + d[:-2])) < 2.5e-4
