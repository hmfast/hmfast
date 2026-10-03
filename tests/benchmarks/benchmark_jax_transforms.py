"""
Compiled counterparts of tests/unit/test_jax_transforms.py.

The unit file only traces each entry point under ``jax.jit``, which is where jittability
failures surface. Here every case is compiled and its jitted numbers are checked against
the eager call, and the end-to-end gradient cases too heavy for the unit suite (halo-model
C_ell through HOD/GNFW profiles, and the halofit Limber spectrum) are checked against finite
differences and between forward and reverse mode.
"""

import jax
import numpy as np
import pytest

from ..unit.test_jax_transforms import (
    CASES,
    GRADIENT_CASES,
    GRADIENT_FD_TOL,
    PARAMS,
    UNIT_GRADIENT_CASES,
    _jacobian_errors,
)

BENCHMARK_GRADIENT_CASES = [name for name in GRADIENT_CASES if name not in UNIT_GRADIENT_CASES]


def _leaves(out):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(out)]


@pytest.mark.parametrize("fn", CASES)
def test_jitted_matches_eager(fn):
    """Runs inside jax.jit, and returns the same finite numbers as the eager call."""
    jitted = _leaves(jax.block_until_ready(jax.jit(fn)(PARAMS)))
    eager = _leaves(jax.block_until_ready(fn(PARAMS)))

    assert len(eager) == len(jitted)

    # Floor taken from the whole output: a term that cancels to zero has no scale of its own.
    scale = max((float(np.max(np.abs(x))) for x in eager if x.size), default=0.0)

    for got, want in zip(jitted, eager):
        assert np.all(np.isfinite(want)), "eager call produced non-finite values"
        assert np.all(np.isfinite(got)), "jitted call produced non-finite values"
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-12 * scale)


@pytest.mark.parametrize("name", BENCHMARK_GRADIENT_CASES)
def test_gradient_matches_finite_difference(name):
    jac, fd_err, _ = _jacobian_errors(name)
    assert np.all(np.isfinite(jac)) and np.all(np.any(jac != 0, axis=0)), "a parameter has no effect or a NaN gradient"
    assert fd_err < GRADIENT_FD_TOL[name]


@pytest.mark.parametrize("name", BENCHMARK_GRADIENT_CASES)
def test_forward_and_reverse_mode_agree(name):
    _, _, mode_err = _jacobian_errors(name)
    assert mode_err < 1e-10  # measured <= 2.0e-14
