# hmfast

**hmfast** is a differentiable, JAX-native halo model code for cosmology. It computes
cosmological observables — power spectra, angular power spectra, higher-order statistics,
and covariances — through the halo model, using neural network emulators in place of a
traditional Boltzmann solver for the background cosmology and the linear matter power
spectrum.

## Features

- **Broad observable support**:
  - Background cosmology and halo-model quantities (`H(z)`, distances, growth factor, halo mass function, halo bias, concentration, ...)
  - 3D and angular power spectra (`P(k,z)`, `C_ℓ`)
  - Halo-model bispectra, trispectra, and covariances
- **Fast** — angular power spectra take tens of milliseconds on CPU, and single-digit
  milliseconds on a powerful GPU.
- **Fully differentiable** — every observable is differentiable end-to-end with respect to
  cosmological and astrophysical parameters via `jax.grad`/`jax.jacobian`, with no
  finite-differencing required.
- **A wide range of tracers and profiles out of the box** — matter (NFW), galaxy clustering
  (HOD), galaxy and CMB lensing, thermal and kinetic Sunyaev-Zel'dovich (tSZ/kSZ), and the
  cosmic infrared background (CIB), all sharing the same halo-model machinery.

## Documentation

Full API documentation: https://hmfast.readthedocs.io

## License

Apache License 2.0 — see `LICENSE`.

## Authors

Patrick Janulewicz, Licong Xu, Boris Bolliet
