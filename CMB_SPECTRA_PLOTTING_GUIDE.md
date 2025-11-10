# CMB Spectra Plotting Guide - Unambiguous Method

## Overview
After porting cl^tt from classy_szfast to hmfast, the plotting method is now **unambiguous**. This document clarifies the correct approach.

## The Unambiguous Method

### For Plotting (Recommended)
Use `get_cmb_spectra()` - returns D_ℓ values ready for direct plotting:

```python
# CORRECT - Unambiguous plotting method
cmb_data = emulator.get_cmb_spectra(params, lmax=2500)
plt.plot(cmb_data['ell'], cmb_data['tt'])  # Direct plotting - NO factors needed!
plt.ylabel('ℓ(ℓ+1)Cℓ^TT/(2π) [μK²]')
```

### For Raw Power Spectra
Use `get_cmb_cls()` - returns raw C_ℓ values:

```python
# For raw power spectra (if needed)
cmb_cls = emulator.get_cmb_cls(params, lmax=2500)
# To plot these, you'd need: cmb_cls['tt'] * ell * (ell+1) / (2*pi)
```

## Key Changes Made

1. **Clear Method Names**:
   - `get_cmb_spectra()` → Returns D_ℓ for plotting (no factors needed)
   - `get_cmb_cls()` → Returns raw C_ℓ (would need factors for plotting)

2. **Unambiguous Documentation**:
   - Method docstrings clearly state what each returns
   - Comments in plotting script emphasize "NO factors needed"

3. **Plotting Script Updates**:
   - Removed ambiguous factor multiplications
   - Added clear comments explaining the direct plotting approach
   - Header documentation explains the unambiguous method

## Migration from classy_szfast

The old classy_szfast approach required manual factor handling which led to confusion. 
The new hmfast approach is:

```python
# OLD (classy_szfast) - ambiguous
cmb = classy_sz.get_cmb_cls(...)
fac = ell * (ell + 1) / (2 * np.pi) 
plt.plot(ell, cmb['tt'] * fac)  # Manual factor - error-prone!

# NEW (hmfast) - unambiguous  
cmb = emulator.get_cmb_spectra(...)
plt.plot(cmb['ell'], cmb['tt'])  # Direct plotting - no ambiguity!
```

## Verification

The plotting script (`scripts/ede_plots.py`) now produces correct spectra with this unambiguous approach. Run the script to verify:

```bash
cd hmfast
python scripts/ede_plots.py
```

All CMB spectra plots (TT, TE, EE, PP) are generated using direct plotting without any manual factor handling.