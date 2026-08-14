"""
CCL benchmark comparisons for hmfast.halos (HMF, bias, concentration, mass
definitions/conversions), used as an external ground truth.

Requires pyccl; the whole module is skipped if it isn't installed. All tests
are marked `ccl` (see pyproject.toml) so `pytest -m "not ccl"` skips this file
without needing pyccl at all.

Tolerances were derived empirically (not guessed): for each component, we ran
this exact comparison at the fixed cosmology below across all 4 mass defs and
z in {0, 0.5, 1.0}, recorded the observed max % residual, and set rtol to a
round number with margin above it (see each test's comment for the number).

Notes on scope, both deliberate:
  - The mass grid here is capped at 1e15: beyond that, dn/dlnM is exponentially suppressed and a tiny
    sigma(M,z) difference between hmfast/CCL blows up in *relative* terms
    (observed up to ~13% at m=1e16, z=2) despite both codes agreeing to
    <1% at m=1e15 -- that's real, benign behavior in the tail, not something
    a fixed tolerance should have to accommodate.
  - z is capped at 1.0: pyccl's own Tinker08 vir-mass-def evaluation raises
    a ValueError for this cosmology at z >~ 1.06 (its internal Delta_mean(z)
    interpolation drops fractionally below its tabulated floor of 200) --
    a CCL-side edge case unrelated to hmfast, sidestepped by not probing it.
"""
import jax.numpy as jnp
import numpy as np
import pytest

pyccl = pytest.importorskip("pyccl")
pytestmark = pytest.mark.ccl

from hmfast.halos.massdef import MassDefinition, mass_translator
from hmfast.halos.massfunc import T08HaloMassFunction, T10HaloMassFunction
from hmfast.halos.bias import T10HaloBias
from hmfast.halos.concentration import D08Concentration, B13Concentration


M_GRID = np.geomspace(1e10, 1e15, 40)
Z_LIST = [0.0, 0.5, 1.0]


def to_ccl_massdef(mass_def):
    reference = "matter" if mass_def.reference == "mean" else mass_def.reference
    return pyccl.halos.MassDef(mass_def.delta, reference)


@pytest.fixture(scope="session")
def cosmo_ccl(fixed_cosmology):
    # A pyccl CosmologyCalculator fed hmfast's own P_lin(k, z), isolating formula
    # differences from any P(k) differences between the two codes' Boltzmann solvers.
    h = fixed_cosmology.H0 / 100.0
    Omega_c = fixed_cosmology.omega_cdm / h**2
    Omega_b = fixed_cosmology.omega_b / h**2

    z_wide = np.linspace(0.0, 3.0, 50)
    a_wide = np.sort(1.0 / (1.0 + z_wide))
    k_ref_jnp, _ = fixed_cosmology._pk_grid()
    k_ref = np.asarray(k_ref_jnp)
    pk_lin = np.asarray(
        fixed_cosmology.pk(k_ref_jnp, jnp.asarray(1.0 / a_wide - 1.0), linear=True)
    ).T  # (Nz, Nk), CCL's expected ordering

    return pyccl.CosmologyCalculator(
        Omega_c=Omega_c, Omega_b=Omega_b, h=h,
        A_s=fixed_cosmology.A_s, n_s=fixed_cosmology.n_s,
        pk_linear={"a": a_wide, "k": k_ref, "delta_matter:delta_matter": pk_lin},
    )


class TestMassDefinitionCCL:
    # r_delta matches CCL's MassDef.get_radius across all mass defs/redshifts (rtol 1%, observed max ~0.1%).
    def test_r_delta_matches_ccl(self, fixed_cosmology, cosmo_ccl, mass_def):
        md_ccl = to_ccl_massdef(mass_def)
        for z in Z_LIST:
            r_hmf = np.asarray(mass_def.r_delta(fixed_cosmology, M_GRID, z)).flatten()
            r_ccl = md_ccl.get_radius(cosmo_ccl, M_GRID, 1.0 / (1.0 + z))
            assert np.allclose(r_hmf, r_ccl, rtol=0.01)


# All (in, out) pairs among {200c, 200m, vir} x {200c, 200m, 500c, vir} with in != out --
# built explicitly (rather than a fixture cross-product + skip) so there's nothing to skip.
MASS_CONVERSION_PAIRS = [
    (in_def, out_def)
    for in_def in [(200, "critical"), (200, "mean"), ("vir", "critical")]
    for out_def in [(200, "critical"), (200, "mean"), (500, "critical"), ("vir", "critical")]
    if in_def != out_def
]


class TestMassConversionCCL:
    # mass_translator matches CCL's mass_translator (D08 concentration) for every non-self def pair (rtol 2%, observed max ~0.18%).
    @pytest.mark.parametrize("in_def,out_def", MASS_CONVERSION_PAIRS)
    def test_translator_matches_ccl(self, fixed_cosmology, cosmo_ccl, in_def, out_def):
        md_in_hmf, md_out_hmf = MassDefinition(*in_def), MassDefinition(*out_def)
        md_in_ccl, md_out_ccl = to_ccl_massdef(md_in_hmf), to_ccl_massdef(md_out_hmf)
        conc_in_ccl = pyccl.halos.concentration.ConcentrationDuffy08(mass_def=md_in_ccl)
        d08 = D08Concentration()

        f_hmf = mass_translator(md_in_hmf, md_out_hmf, d08)
        f_ccl = pyccl.halos.massdef.mass_translator(mass_in=md_in_ccl, mass_out=md_out_ccl, concentration=conc_in_ccl)

        for z in Z_LIST:
            m_hmf = np.asarray(f_hmf(fixed_cosmology, M_GRID, z)).flatten()
            m_ccl = np.asarray(f_ccl(cosmo_ccl, M_GRID, 1.0 / (1.0 + z)))
            assert np.allclose(m_hmf, m_ccl, rtol=0.02)


class TestHaloMassFunctionCCL:
    # T08/T10 dndlnm matches CCL's MassFuncTinker08/10 (rtol 3%/5%, observed max ~0.96%/1.63%).
    @pytest.mark.parametrize("hmf_cls,ccl_hmf_cls,rtol", [
        (T08HaloMassFunction, "MassFuncTinker08", 0.03),
        (T10HaloMassFunction, "MassFuncTinker10", 0.05),
    ])
    def test_dndlnm_matches_ccl(self, fixed_cosmology, cosmo_ccl, mass_def, hmf_cls, ccl_hmf_cls, rtol):
        md_ccl = to_ccl_massdef(mass_def)
        hmf_ccl = getattr(pyccl.halos.hmfunc, ccl_hmf_cls)(mass_def=md_ccl, mass_def_strict=True)
        for z in Z_LIST:
            y_hmf = np.asarray(hmf_cls().dndlnm(fixed_cosmology, M_GRID, z, mass_def=mass_def))
            y_ccl = hmf_ccl(cosmo_ccl, M_GRID, 1.0 / (1.0 + z)) / np.log(10)
            assert np.allclose(y_hmf, y_ccl, rtol=rtol)


class TestHaloBiasCCL:
    # T10HaloBias(order=1) matches CCL's HaloBiasTinker10 (rtol 1%, observed max ~0.05%).
    def test_bias_matches_ccl(self, fixed_cosmology, cosmo_ccl, mass_def):
        md_ccl = to_ccl_massdef(mass_def)
        bias_ccl = pyccl.halos.hbias.HaloBiasTinker10(mass_def=md_ccl)
        for z in Z_LIST:
            b_hmf = np.asarray(T10HaloBias().bias(fixed_cosmology, M_GRID, z, mass_def=mass_def, order=1))
            b_ccl = bias_ccl(cosmo_ccl, M_GRID, 1.0 / (1.0 + z))
            assert np.allclose(b_hmf, b_ccl, rtol=0.01)


class TestConcentrationCCL:
    # D08/B13 concentration matches CCL's Duffy08/Bhattacharya13 at their 3 shared supported defs (rtol 0.1%/1%, observed max ~0%/0.22%).
    @pytest.mark.parametrize("conc_cls,ccl_conc_cls,rtol", [
        (D08Concentration, "ConcentrationDuffy08", 0.001),
        (B13Concentration, "ConcentrationBhattacharya13", 0.01),
    ])
    @pytest.mark.parametrize("delta,reference", [(200, "critical"), (200, "mean"), ("vir", "critical")])
    def test_concentration_matches_ccl(self, fixed_cosmology, cosmo_ccl, conc_cls, ccl_conc_cls, rtol, delta, reference):
        md_hmf = MassDefinition(delta, reference)
        md_ccl = to_ccl_massdef(md_hmf)
        conc_ccl = getattr(pyccl.halos.concentration, ccl_conc_cls)(mass_def=md_ccl)
        for z in Z_LIST:
            c_hmf = np.asarray(conc_cls().c_delta(fixed_cosmology, M_GRID, z, mass_def=md_hmf))
            c_ccl = conc_ccl(cosmo_ccl, M_GRID, 1.0 / (1.0 + z))
            assert np.allclose(c_hmf, c_ccl, rtol=rtol)
