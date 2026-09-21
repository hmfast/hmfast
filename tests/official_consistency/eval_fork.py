"""Evaluate overlapping hmfast quantities with the licongxu fork API."""

from __future__ import annotations

import os
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
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile, NFWMatterProfile
from hmfast.tracers import (
    CMBLensingTracer,
    CIBTracer,
    GalaxyHODTracer,
    GalaxyLensingTracer,
    kSZTracer,
    tSZTracer,
)
from hmfast.utils import log_interp1d_extrap


def _np(x):
    return np.asarray(jax.device_get(x), dtype=np.float64)


def main(out_path):
    cosmo = Cosmology(
        emulator_set="lcdm:v1",
        H0=68.0,
        omega_cdm=0.12,
        omega_b=0.02246576,
        ln1e10A_s=LN10AS,
        n_s=0.965,
        tau_reio=0.0544,
    )
    hm = HaloModel(cosmology=cosmo)
    matter = NFWMatterProfile()
    gnfw = GNFWPressureProfile()
    cmb = CMBLensingTracer(profile=matter)
    tsz = tSZTracer(profile=gnfw)
    ksz = kSZTracer()
    gal = GalaxyHODTracer()
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
    k_grid, _ = cosmo.pk(0.0, linear=True)
    pk_lin = np.stack(
        [_np(log_interp1d_extrap(k_pk, *cosmo.pk(float(z), linear=True))) for z in Z_PK],
        axis=1,
    )
    pk_nl = np.stack(
        [_np(log_interp1d_extrap(k_pk, *cosmo.pk(float(z), linear=False))) for z in Z_PK],
        axis=1,
    )

    ell_tt, cl_tt = cosmo.cl_tt()
    ell_ee, cl_ee = cosmo.cl_ee()
    ell_te, cl_te = cosmo.cl_te()
    ell_pp, cl_pp = cosmo.cl_pp()
    cl_tt_i = _np(jnp.interp(ell_cmb, ell_tt, cl_tt))
    cl_ee_i = _np(jnp.interp(ell_cmb, ell_ee, cl_ee))
    cl_te_i = _np(jnp.interp(ell_cmb, ell_te, cl_te))
    cl_pp_i = _np(jnp.interp(ell_cmb, ell_pp, cl_pp))

    chi = _np(cosmo.angular_diameter_distance(z_cl) * (1.0 + z_cl))

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
        "cl_tt": cl_tt_i,
        "cl_ee": cl_ee_i,
        "cl_te": cl_te_i,
        "cl_pp": cl_pp_i,
        "r_delta": _np(hm.mass_definition.r_delta(cosmo, m_halo, z_prof)),
        "hmf": _np(hm.halo_mass_function.halo_mass_function(hm, m_halo, z_prof)),
        "bias": _np(hm.halo_bias.halo_bias(hm, m_halo, z_prof, order=1)),
        "conc": _np(hm.concentration.c_delta(hm, m_halo, z_prof)),
        "nfw_ur": _np(matter.u_r(hm, r_prof, m_prof, z_prof)),
        "nfw_uk": _np(matter.u_k(hm, k_prof, m_prof, z_prof)),
        "gnfw_ur": _np(gnfw.u_r(hm, r_prof, m_prof, z_prof)),
        "gnfw_uk": _np(gnfw.u_k(hm, k_prof, m_prof, z_prof)),
        "kernel_cmb": _np(cmb.kernel(cosmo, z_cl)),
        "kernel_tsz": _np(tsz.kernel(cosmo, z_cl)),
        "kernel_ksz": _np(ksz.kernel(cosmo, z_cl)),
        "kernel_gal": _np(gal.kernel(cosmo, z_cl)),
        "kernel_glens": _np(glens.kernel(cosmo, z_cl)),
        "kernel_cib": _np(cib.kernel(cosmo, z_cl)),
        "chi_cl": chi,
        "pk1h_matter": _np(hm.pk_1h(cmb, cmb, k_pk, M_HALO, jnp.array([0.5]))),
        "pk2h_matter": _np(hm.pk_2h(cmb, cmb, k_pk, M_HALO, jnp.array([0.5]))),
        "cl1h_matter": _np(hm.cl_1h(cmb, cmb, ell, M_HALO, z_cl)),
        "cl2h_matter": _np(hm.cl_2h(cmb, cmb, ell, M_HALO, z_cl)),
        "pk1h_tsz": _np(hm.pk_1h(tsz, tsz, k_pk, M_HALO, jnp.array([0.5]))),
        "cl1h_tsz": _np(hm.cl_1h(tsz, tsz, ell, M_HALO, z_cl)),
        "cl2h_tsz": _np(hm.cl_2h(tsz, tsz, ell, M_HALO, z_cl)),
        "As": np.array(AS),
        "ln10As": np.array(LN10AS),
    }
    np.savez_compressed(out_path, **out)
    print(f"wrote {out_path} with {len(out)} arrays")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fork_hmfast.npz"
    main(out)
