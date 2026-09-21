"""Evaluate overlapping hmfast quantities with the official hmfast API."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from grids import (
    AS,
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
from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel, Pk
from hmfast.halos.profiles import GNFWPressureProfile, NFWMatterProfile
from hmfast.tracers import (
    CMBLensingTracer,
    CIBTracer,
    GalaxyLensingTracer,
    GalaxyTracer,
    kSZTracer,
    tSZTracer,
)


def _np(x):
    return np.asarray(jax.device_get(x), dtype=np.float64)


def _reshape_pk(arr, nk, nz):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return np.full((nk, nz), arr)
    if arr.ndim == 1:
        if arr.shape[0] == nk:
            return arr[:, None]
        if arr.shape[0] == nz:
            return arr[None, :]
    return arr.reshape(nk, nz)


def main(out_path):
    cosmo = Cosmology(
        emulator_set="lcdm:v1",
        H0=68.0,
        omega_cdm=0.12,
        omega_b=0.02246576,
        A_s=AS,
        n_s=0.965,
        tau=0.0544,
    )
    hm = HaloModel(cosmology=cosmo, m_grid=jnp.asarray(M_HALO))
    stats = Pk()
    matter = NFWMatterProfile()
    gnfw = GNFWPressureProfile()
    cmb = CMBLensingTracer(profile=matter)
    tsz = tSZTracer(profile=gnfw)
    ksz = kSZTracer()
    gal = GalaxyTracer()
    glens = GalaxyLensingTracer()
    cib = CIBTracer()

    z_bg = jnp.asarray(Z_BG)
    z_pk = jnp.asarray(Z_PK)
    k_pk = jnp.asarray(K_PK)
    m_halo = jnp.asarray(M_HALO)
    m_prof = jnp.asarray(M_PROF)
    r_prof = jnp.asarray(R_PROF)
    k_prof = jnp.asarray(K_PROF)
    z_prof = jnp.asarray(Z_PROF)
    ell = jnp.asarray(ELL)
    z_cl = jnp.asarray(Z_CL)
    ell_cmb = jnp.asarray(ELL_CMB)

    derived = cosmo.derived_parameters()
    k_grid, _ = cosmo._pk_grid()
    pk_lin = _reshape_pk(_np(cosmo.pk(k_pk, z_pk, linear=True)), len(K_PK), len(Z_PK))
    pk_nl = _reshape_pk(_np(cosmo.pk(k_pk, z_pk, linear=False)), len(K_PK), len(Z_PK))

    cl_tt_i = _np(cosmo.cl("TT", ell_cmb))
    cl_ee_i = _np(cosmo.cl("EE", ell_cmb))
    cl_te_i = _np(cosmo.cl("TE", ell_cmb))
    cl_pp_i = _np(cosmo.cl("PP", ell_cmb))

    chi = _np(cosmo.angular_diameter_distance(z_cl) * (1.0 + z_cl))
    mass_def = hm.mass_def

    pk1h_matter = _np(stats.pk_1h(hm, k_pk, jnp.array([0.5]), matter))
    pk2h_matter = _np(stats.pk_2h(hm, k_pk, jnp.array([0.5]), matter))
    pk1h_tsz = _np(stats.pk_1h(hm, k_pk, jnp.array([0.5]), gnfw))

    out = {
        "k_grid": _np(k_grid),
        "H": _np(cosmo.hubble_parameter(z_bg)),
        "DA": _np(cosmo.angular_diameter_distance(z_bg)),
        "rho_crit": _np(cosmo.critical_density(z_bg)),
        "omega_m": _np(cosmo.omega_m(z_bg)),
        "growth": _np(cosmo.growth_factor(z_bg)),
        "growth_rate": _np(cosmo.growth_rate(z_bg)),
        "vdisp": _np(cosmo.velocity_dispersion(z_bg)),
        "dV": _np(cosmo.comoving_volume_element(z_bg)),
        "sigma8": _np(cosmo.sigma8(z_bg)),
        "derived_100theta": _np(derived["100*theta_s"]),
        "derived_sigma8": _np(derived["sigma8"]),
        "derived_chi_star": _np(derived["chi_star"]),
        "pk_lin": pk_lin,
        "pk_nl": pk_nl,
        "cl_tt": np.atleast_1d(cl_tt_i),
        "cl_ee": np.atleast_1d(cl_ee_i),
        "cl_te": np.atleast_1d(cl_te_i),
        "cl_pp": np.atleast_1d(cl_pp_i),
        "r_delta": _np(mass_def.r_delta(cosmo, m_halo, z_prof)),
        "hmf": _np(hm.halo_mass_function.dndlnm(cosmo, m_halo, z_prof, mass_def)),
        "bias": _np(hm.halo_bias.bias(cosmo, m_halo, z_prof, mass_def, order=1)),
        "conc": _np(hm.concentration.c_delta(cosmo, m_halo, z_prof, mass_def)),
        "nfw_ur": _np(matter.real(hm, r_prof, m_prof, z_prof)),
        "nfw_uk": _np(matter.fourier(hm, k_prof, m_prof, z_prof)),
        "gnfw_ur": _np(gnfw.real(hm, r_prof, m_prof, z_prof)),
        "gnfw_uk": _np(gnfw.fourier(hm, k_prof, m_prof, z_prof)),
        "kernel_cmb": _np(cmb.kernel(cosmo, z_cl)),
        "kernel_tsz": _np(tsz.kernel(cosmo, z_cl)),
        "kernel_ksz": _np(ksz.kernel(cosmo, z_cl)),
        "kernel_gal": _np(gal.kernel(cosmo, z_cl)),
        "kernel_glens": _np(glens.kernel(cosmo, z_cl)),
        "kernel_cib": _np(cib.kernel(cosmo, z_cl)),
        "chi_cl": chi,
        "pk1h_matter": np.atleast_2d(pk1h_matter.reshape(len(K_PK), -1)),
        "pk2h_matter": np.atleast_2d(pk2h_matter.reshape(len(K_PK), -1)),
        "cl1h_matter": np.atleast_1d(_np(stats.cl_1h(hm, cmb, cmb, ell, z_cl))),
        "cl2h_matter": np.atleast_1d(_np(stats.cl_2h(hm, cmb, cmb, ell, z_cl))),
        "pk1h_tsz": np.atleast_2d(pk1h_tsz.reshape(len(K_PK), -1)),
        "cl1h_tsz": np.atleast_1d(_np(stats.cl_1h(hm, tsz, tsz, ell, z_cl))),
        "cl2h_tsz": np.atleast_1d(_np(stats.cl_2h(hm, tsz, tsz, ell, z_cl))),
        "As": np.array(AS),
        "ln10As": np.array(LN10AS),
    }
    np.savez_compressed(out_path, **out)
    print(f"wrote {out_path} with {len(out)} arrays")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/official_hmfast.npz"
    main(out)
