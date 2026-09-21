"""Compare fork vs official hmfast evaluation dumps."""

from __future__ import annotations

import json
import sys

import numpy as np

KERNEL_KEYS = {
    "kernel_cmb",
    "kernel_tsz",
    "kernel_ksz",
    "kernel_gal",
    "kernel_glens",
    "kernel_cib",
}


def rel_stats(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        try:
            b = b.reshape(a.shape)
        except ValueError:
            return {
                "shape_a": list(a.shape),
                "shape_b": list(b.shape),
                "error": "shape mismatch",
            }
    finite = np.isfinite(a) & np.isfinite(b)
    n = int(finite.sum())
    if n == 0:
        return {"n_finite": 0, "error": "no finite values"}
    aa = a[finite]
    bb = b[finite]
    scale = np.maximum(np.maximum(np.abs(aa), np.abs(bb)), 1e-30)
    rel = np.abs(aa - bb) / scale
    absdiff = np.abs(aa - bb)
    med_ratio = float(np.median(aa / np.where(np.abs(bb) < 1e-30, np.nan, bb)))
    return {
        "n_finite": n,
        "max_rel": float(np.max(rel)),
        "median_rel": float(np.median(rel)),
        "p95_rel": float(np.percentile(rel, 95)),
        "max_abs": float(np.max(absdiff)),
        "median_ratio_fork_over_official": med_ratio,
        "rms_rel": float(np.sqrt(np.mean(rel**2))),
    }


def main(fork_path, official_path, report_path=None):
    fork = np.load(fork_path)
    official = np.load(official_path)
    keys = sorted(set(fork.files) & set(official.files))
    report = {}
    print(f"{'quantity':28s} {'max_rel':>10s} {'p95_rel':>10s} {'median_rel':>12s} {'median_ratio':>14s}")
    for key in keys:
        stats = rel_stats(fork[key], official[key])
        report[key] = stats
        if "error" in stats:
            print(f"{key:28s} ERROR {stats}")
            continue
        print(
            f"{key:28s} {stats['max_rel']:10.3e} {stats['p95_rel']:10.3e} "
            f"{stats['median_rel']:12.3e} {stats['median_ratio_fork_over_official']:14.4g}"
        )

    # Kernel convention check: fork kernels should be official / chi^2 for Limber-W(chi) tracers.
    chi = official["chi_cl"]
    print("\nKernel convention check (fork * chi^2 vs official):")
    for key in sorted(KERNEL_KEYS):
        if key not in fork.files or key not in official.files:
            continue
        fork_wchi = fork[key] * chi**2
        stats = rel_stats(fork_wchi, official[key])
        report[f"{key}_times_chi2"] = stats
        print(
            f"{key:28s} {stats['max_rel']:10.3e} {stats['median_rel']:12.3e} "
            f"ratio={stats['median_ratio_fork_over_official']:10.4g}"
        )

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {report_path}")
    return report


if __name__ == "__main__":
    fork_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fork_hmfast.npz"
    official_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/official_hmfast.npz"
    report_path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/hmfast_compare.json"
    main(fork_path, official_path, report_path)
