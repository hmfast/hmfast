The halo model module
=====================

This page documents the primary public functions provided by the ``HaloModel`` class in ``hmfast``. It covers the following functions:

- ``get_hmf(z, m, params)`` — Returns the halo mass function :math:`\mathrm{d}n/\mathrm{d}\ln M` evaluated at redshift ``z`` and halo mass array ``m``.  
- ``get_hbf(z, m, params)`` — Returns the halo bias function :math:`b(m)` evaluated at redshift ``z`` and mass array ``m``.  
- ``get_C_ell_1h(tracer, z, m, ell, params)`` — Computes the 1-halo contribution to the angular power spectrum :math:`C_\ell` for a given tracer.  
- ``get_C_ell_2h(tracer, z, m, ell, params)`` — Computes the 2-halo contribution to the angular power spectrum :math:`C_\ell` for a given tracer.  



Setting up your halo model
--------------------------

To use the ``HaloModel`` class, you must first instantiate your cosmological parameters and load the emulator. Here's an example:

.. code-block:: python

    import hmfast 
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    
    
    params_hmfast = {
        'omega_b': 0.02,
        'omega_cdm':  0.12,
        'H0': 67.5, 
        'tau_reio': 0.0561,
        'ln10^{10}A_s': 3.047,
        'n_s': 0.9665
    }


.. code-block:: python

    
    # Define the halo model
    halo_model = hmfast.halo_model.HaloModel(cosmo_model=0)

    # Mass and redshift grids
    z_grid = jnp.geomspace(0.005, 3.0, 100)
    m_grid = jnp.geomspace(5e10, 3.5e15, 100)



Halo mass and bias functions
----------------------------

With our halo model defined, we can now compute halo mass and bias functions.

.. code-block:: python

   # Redshifts and mass grid
   z0, z1, z2 = 0.0, 1.0, 2.0

   # --- Compute halo mass function ---
   hmf_z0 = halo_model.get_hmf(z0, m_grid, params=params_hmfast)
   hmf_z1 = halo_model.get_hmf(z1, m_grid, params=params_hmfast)
   hmf_z2 = halo_model.get_hmf(z2, m_grid, params=params_hmfast)

   # --- Compute halo bias function ---
   hbf_z0 = halo_model.get_hbf(z0, m_grid, params=params_hmfast)
   hbf_z1 = halo_model.get_hbf(z1, m_grid, params=params_hmfast)
   hbf_z2 = halo_model.get_hbf(z2, m_grid, params=params_hmfast)

   # --- Plot the results ---
   z_values = [z0, z1, z2]
   hmf_results = [hmf_z0, hmf_z1, hmf_z2]
   hbf_results = [hbf_z0, hbf_z1, hbf_z2]

   base_color, linestyles, alphas = "C0", ["-", "--", "-."], [1.0, 0.75, 0.55]
   fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

   # Plot halo mass function
   for z, hmf, ls, a in zip(z_values, hmf_results, linestyles, alphas):
       axes[0].loglog(m_grid, hmf, linestyle=ls, color=base_color, alpha=a, label=f"z = {z}")
   axes[0].set_ylabel(r"$\mathrm{d}n/\mathrm{d}\ln M$")
   axes[0].set_ylim(1e-6, 1e-1)   # ← the single added line
   axes[0].legend()

   # Plot halo bias function
   for z, hbf, ls, a in zip(z_values, hbf_results, linestyles, alphas):
       axes[1].loglog(m_grid, hbf, linestyle=ls, color=base_color, alpha=a, label=f"z = {z}")
   axes[1].set_xlabel(r"$M\ [M_\odot/h]$")
   axes[1].set_ylabel(r"$b(M)$")
   axes[1].legend()

   plt.tight_layout()
   plt.show()


.. image:: _static/hmf_hbf.png
   :width: 90%
   :align: center
   :alt: Halo mass and bias functions


Adding tracers
---------------

Once you are ready to compute angular power spectra, you can create tracer objects via the ``add_tracer`` method. Each tracer evaluates a profile over a dimensionless radial grid:

.. math::

    x = \frac{r}{r_{\rm scale}}

For tSZ tracers, the scale is the radius :math:`r_{\Delta}` (the radius enclosing :math:`\Delta` times the critical density).
For galaxy HOD tracers, the scale is the halo scale radius ``r_s`` from the NFW profile.  

This ``x_grid`` is a defining property of the tracer that cannot be changed after creation.  
If you wish to use a different radial grid, simply create a new tracer with a new ``x_grid``.


.. code-block:: python

    # Define radial grids for tracers
    x_grid_tsz = jnp.logspace(jnp.log10(1e-4), jnp.log10(20.0), 512)
    x_grid_hod = jnp.logspace(jnp.log10(1e-5), jnp.log10(50.0), 512)
    
    # Add tracers
    tsz_tracer = halo_model.create_tracer("y", x=x_grid_tsz)
    galaxy_hod_tracer = halo_model.create_tracer("g", x=x_grid_hod)



Angular power spectra
---------------------

You may now easily compute the 1-halo and 2-halo of your tSZ tracer:


.. code-block:: python

   # --- Define ell grid and compute 1 halo and 2 halo angular power spectrum  ---
   ell_grid_tsz = jnp.geomspace(2, 8e3, 50)
   C_ell_yy_1h = halo_model.get_C_ell_1h(tsz_tracer, z_grid, m_grid, ell_grid_tsz, params=params_hmfast)
   C_ell_yy_2h = halo_model.get_C_ell_2h(tsz_tracer, z_grid, m_grid, ell_grid_tsz, params=params_hmfast)

   # --- Convert to D_ell ---
   D_ell_yy_1h = ell_grid_tsz * (ell_grid_tsz + 1) * C_ell_yy_1h / (2 * jnp.pi) * 1e12
   D_ell_yy_2h = ell_grid_tsz * (ell_grid_tsz + 1) * C_ell_yy_2h / (2 * jnp.pi) * 1e12

   # --- Plot the results ---
   plt.figure()
   plt.loglog(ell_grid_tsz, D_ell_yy_1h, label="1-halo term")
   plt.loglog(ell_grid_tsz, D_ell_yy_2h, label="2-halo term")
   plt.xlabel(r"$\ell$")
   plt.ylabel(r"$10^{12} D_\ell$")
   plt.legend()
   plt.grid(True, which="both", linestyle="--", alpha=0.5)
   plt.show()


.. image:: _static/C_ell_yy.png
   :width: 90%
   :align: center
   :alt: tSZ angular power spectrum


And similarly for your galaxy HOD tracer:


.. code-block:: python

    # --- Define ell grid and compute 1 halo and 2 halo angular power spectrum  ---
   ell_grid_hod = jnp.geomspace(1e2, 3.5e3, 50)
   C_ell_gg_1h = halo_model.get_C_ell_1h(galaxy_hod_tracer, z_grid, m_grid, ell_grid_hod, params=params_hmfast)
   C_ell_gg_2h = halo_model.get_C_ell_2h(galaxy_hod_tracer, z_grid, m_grid, ell_grid_hod, params=params_hmfast)


   # --- Plot the results ---
   plt.figure()
   plt.loglog(ell_grid_tsz, C_ell_gg_1h, label="1-halo term")
   plt.loglog(ell_grid_tsz, C_ell_gg_2h, label="2-halo term")
   plt.xlabel(r"$\ell$")
   plt.ylabel(r"$10^{12} D_\ell$")
   plt.legend()
   plt.grid(True, which="both", linestyle="--", alpha=0.5)
   plt.show()

.. image:: _static/C_ell_gg.png
   :width: 90%
   :align: center
   :alt: Galaxy HOD angular power spectrum
