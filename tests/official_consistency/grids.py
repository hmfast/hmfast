"""Shared evaluation grids for fork vs official hmfast comparisons."""

import numpy as np

# Match both packages on the fork's default LCDM emulator inputs.
LN10AS = 3.035173309489548
AS = float(np.exp(LN10AS) / 1e10)

Z_BG = np.array([0.0, 0.2, 0.5, 1.0, 2.0, 3.0])
Z_PK = np.array([0.0, 0.5, 1.0])
K_PK = np.geomspace(1e-3, 10.0, 32)
M_HALO = np.geomspace(1e10, 1e15, 48)
M_PROF = np.geomspace(1e12, 1e15, 8)
R_PROF = np.geomspace(1e-2, 5.0, 16)
K_PROF = np.geomspace(1e-2, 5.0, 16)
Z_PROF = np.array([0.3, 0.8])
ELL = np.array([50.0, 200.0, 500.0, 1000.0, 2000.0])
Z_CL = np.linspace(0.05, 3.0, 24)
ELL_CMB = np.array([2, 10, 50, 100, 500, 1000, 2000], dtype=np.float64)
