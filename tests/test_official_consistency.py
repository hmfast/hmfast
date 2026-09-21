"""Numerical consistency of overlapping hmfast quantities vs official hmfast.

The licongxu fork keeps a different public API from
https://github.com/hmfast/hmfast (``u_r``/``u_k`` vs ``real``/``fourier``,
``pk_1h`` on ``HaloModel`` vs ``stats.Pk``, Limber kernels as
:math:`W(\\chi)/\\chi^2`, ...). This test freezes official outputs at
commit ``243f744`` and a fixed cosmology and checks that shared physics
still matches.

Known, documented differences that are *not* regressions:

- tSZ / kSZ Fourier profiles in this fork are projected :math:`u_\\ell`
  (tszpower-style). Official uses 3D Fourier profiles plus
  :math:`dV/\\chi^4` Limber weights.
- ``GNFWPressureProfile`` converts the halo mass to :math:`500c` and uses
  the Arnaud X-ray :math:`h_{70}^{-3/2}` normalization. Official evaluates
  the native ``mass_def`` (default :math:`200c`) with ``P0_hexp=-1``.
- Halo-mass-function interpolation is log-log in this fork vs interpolating
  :math:`\\sigma(M)` then evaluating :math:`f(\\sigma)` in official.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.profiles import NFWMatterProfile
from hmfast.tracers import CMBLensingTracer, CIBTracer, GalaxyHODTracer, GalaxyLensingTracer
from hmfast.utils import log_interp1d_extrap

from tests.official_consistency.grids import (
    ELL,
    ELL_CMB,
    K_PK,
    K_PROF,
    LN10AS,
    M_HALO,
    M_PROF,
    R_PROF,
    Z_BG,
    Z_CL,
    Z_PK,
    Z_PROF,
)


REF_PATH = Path(__file__).parent / "official_consistency" / "reference_official.npz"


def _np(x):
    return np.asarray(jax.device_get(x), dtype=np.float64)


def _max_rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        b = np.reshape(b, a.shape)
    finite = np.isfinite(a) & np.isfinite(b)
    scale = np.maximum(np.maximum(np.abs(a[finite]), np.abs(b[finite])), 1e-30)
    return float(np.max(np.abs(a[finite] - b[finite]) / scale))


@pytest.fixture(scope="module")
def official_ref():
    if not REF_PATH.exists():
        pytest.skip(f"missing official reference dump {REF_PATH}")
    return np.load(REF_PATH)


@pytest.fixture(scope="module")
def cosmology():
    return Cosmology(
        emulator_set="lcdm:v1",
        H0=68.0,
        omega_cdm=0.12,
        omega_b=0.02246576,
        ln1e10A_s=LN10AS,
        n_s=0.965,
        tau_reio=0.0544,
    )


@pytest.fixture(scope="module")
def halo_model(cosmology):
    return HaloModel(cosmology=cosmology)


class TestOfficialCosmology:
    def test_background(self, cosmology, official_ref):
        z = jnp.asarray(Z_BG)
        assert _max_rel(cosmology.hubble_parameter(z), official_ref["H"]) < 1e-12
        assert _max_rel(cosmology.angular_diameter_distance(z), official_ref["DA"]) < 1e-12
        assert _max_rel(cosmology.critical_density(z), official_ref["rho_crit"]) < 1e-12
        assert _max_rel(cosmology.omega_m(z), official_ref["omega_m"]) < 1e-12
        assert _max_rel(cosmology.comoving_volume_element(z), official_ref["dV"]) < 1e-12

    def test_growth_and_pk(self, cosmology, official_ref):
        z = jnp.asarray(Z_BG)
        k = jnp.asarray(K_PK)
        assert _max_rel(cosmology.growth_factor(z), official_ref["growth"]) < 1e-12
        assert _max_rel(cosmology.growth_rate(z), official_ref["growth_rate"]) < 1e-12
        assert _max_rel(cosmology.velocity_dispersion(z), official_ref["vdisp"]) < 1e-12
        assert _max_rel(cosmology.sigma8(z), official_ref["sigma8"]) < 1e-12
        pk_lin = np.stack(
            [_np(log_interp1d_extrap(k, *cosmology.pk(float(zi), linear=True))) for zi in Z_PK],
            axis=1,
        )
        pk_nl = np.stack(
            [_np(log_interp1d_extrap(k, *cosmology.pk(float(zi), linear=False))) for zi in Z_PK],
            axis=1,
        )
        assert _max_rel(pk_lin, official_ref["pk_lin"]) < 1e-12
        assert _max_rel(pk_nl, official_ref["pk_nl"]) < 1e-12

    def test_cmb_spectra(self, cosmology, official_ref):
        ell = jnp.asarray(ELL_CMB)
        ell_tt, cl_tt = cosmology.cl_tt()
        ell_ee, cl_ee = cosmology.cl_ee()
        ell_te, cl_te = cosmology.cl_te()
        ell_pp, cl_pp = cosmology.cl_pp()
        assert _max_rel(jnp.interp(ell, ell_tt, cl_tt), official_ref["cl_tt"]) < 1e-12
        assert _max_rel(jnp.interp(ell, ell_ee, cl_ee), official_ref["cl_ee"]) < 1e-12
        assert _max_rel(jnp.interp(ell, ell_te, cl_te), official_ref["cl_te"]) < 1e-12
        assert _max_rel(jnp.interp(ell, ell_pp, cl_pp), official_ref["cl_pp"]) < 1e-12

    def test_derived_parameters(self, cosmology, official_ref):
        derived = cosmology.derived_parameters()
        assert _max_rel(derived["100*theta_s"], official_ref["derived_100theta"]) < 1e-12
        assert _max_rel(derived["sigma8"], official_ref["derived_sigma8"]) < 1e-12
        assert _max_rel(derived["chi_star"], official_ref["derived_chi_star"]) < 1e-12


class TestOfficialHaloIngredients:
    def test_radius_concentration_nfw(self, halo_model, official_ref):
        m = jnp.asarray(M_HALO)
        z = jnp.asarray(Z_PROF)
        matter = NFWMatterProfile()
        r = jnp.asarray(R_PROF)
        k = jnp.asarray(K_PROF)
        m_p = jnp.asarray(M_PROF)
        assert _max_rel(halo_model.mass_definition.r_delta(halo_model.cosmology, m, z), official_ref["r_delta"]) < 1e-12
        assert _max_rel(halo_model.concentration.c_delta(halo_model, m, z), official_ref["conc"]) < 1e-12
        assert _max_rel(matter.u_k(halo_model, k, m_p, z), official_ref["nfw_uk"]) < 1e-12
        nfw_ur = _np(matter.u_r(halo_model, r, m_p, z))
        # Official truncates NFW at r_delta; compare only inside the halo.
        inside = official_ref["nfw_ur"] != 0.0
        assert _max_rel(nfw_ur[inside], official_ref["nfw_ur"][inside]) < 1e-12
        assert np.allclose(nfw_ur[~inside], 0.0)

    def test_hmf_and_bias(self, halo_model, official_ref):
        m = jnp.asarray(M_HALO)
        z = jnp.asarray(Z_PROF)
        # Log-log HMF interpolation vs official sigma-grid interpolation.
        assert _max_rel(halo_model.halo_mass_function.halo_mass_function(halo_model, m, z), official_ref["hmf"]) < 2e-3
        assert _max_rel(halo_model.halo_bias.halo_bias(halo_model, m, z, order=1), official_ref["bias"]) < 2e-3


class TestOfficialMatterSpectra:
    def test_pk_and_cl_matter(self, halo_model, official_ref):
        tracer = CMBLensingTracer(profile=NFWMatterProfile())
        k = jnp.asarray(K_PK)
        m = jnp.asarray(M_HALO)
        z1 = jnp.array([0.5])
        ell = jnp.asarray(ELL)
        z_cl = jnp.asarray(Z_CL)
        pk1h = _np(halo_model.pk_1h(tracer, tracer, k, m, z1))
        pk2h = _np(halo_model.pk_2h(tracer, tracer, k, m, z1))
        cl1h = _np(halo_model.cl_1h(tracer, tracer, ell, m, z_cl))
        cl2h = _np(halo_model.cl_2h(tracer, tracer, ell, m, z_cl))
        assert _max_rel(pk1h, official_ref["pk1h_matter"]) < 1e-3
        assert _max_rel(pk2h, official_ref["pk2h_matter"]) < 1e-3
        assert _max_rel(cl1h, official_ref["cl1h_matter"]) < 1e-3
        assert _max_rel(cl2h, official_ref["cl2h_matter"]) < 1e-3


class TestOfficialKernelConvention:
    def test_limber_kernels_are_official_over_chi2(self, cosmology, official_ref):
        z = jnp.asarray(Z_CL)
        chi = _np(cosmology.angular_diameter_distance(z) * (1.0 + z))
        cmb = CMBLensingTracer()
        gal = GalaxyHODTracer()
        glens = GalaxyLensingTracer()
        cib = CIBTracer()
        assert _max_rel(_np(cmb.kernel(cosmology, z)) * chi**2, official_ref["kernel_cmb"]) < 1e-12
        assert _max_rel(_np(gal.kernel(cosmology, z)) * chi**2, official_ref["kernel_gal"]) < 1e-12
        fork_glens = _np(glens.kernel(cosmology, z)) * chi**2
        off_glens = official_ref["kernel_glens"]
        finite = np.isfinite(fork_glens) & np.isfinite(off_glens)
        assert _max_rel(fork_glens[finite], off_glens[finite]) < 1e-12
        assert _max_rel(_np(cib.kernel(cosmology, z)) * chi**2, official_ref["kernel_cib"]) < 1e-12
