The halo model module
=====================

This page documents the primary public functions provided by the ``HaloModel`` class in ``hmfast``. It covers the following functions:

- ``halo_mass_function(z, m, params)`` — Returns the halo mass function :math:`\mathrm{d}n/\mathrm{d}\ln M` evaluated at redshift ``z`` and halo mass array ``m``.  
- ``halo_bias_function(z, m, params)`` — Returns the halo bias function :math:`b(m)` evaluated at redshift ``z`` and mass array ``m``.  
- ``cl_1h(tracer, z, m, l, params)`` — Computes the 1-halo contribution to the angular power spectrum :math:`C_\ell` for a given tracer.  
- ``cl_2h(tracer, z, m, l, params)`` — Computes the 2-halo contribution to the angular power spectrum :math:`C_\ell` for a given tracer.  



Setting up your halo model
--------------------------

To use the ``HaloModel`` class, you must first instantiate your halo model by specifying which emulators you wish to use. Here's an example:

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

   # Define the halo model instance
    halo_model = hmfast.halo_model.HaloModel(cosmo_model=0)



Halo mass function
----------------------------

With our halo model defined, we can now compute halo mass function. 
In this example, we will evaluate three scalar redshifts for an array of masses.
However, you can pass any configuration of arrays/scalars for mass and redshift.

.. code-block:: python

   hmf_z0 = halo_model.halo_mass_function(0.0, m=jnp.geomspace(5e10, 3.5e15, 100), params=params_hmfast)
   hmf_z1 = halo_model.halo_mass_function(1.0, m=jnp.geomspace(5e10, 3.5e15, 100), params=params_hmfast)
   hmf_z2 = halo_model.halo_mass_function(2.0, m=jnp.geomspace(5e10, 3.5e15, 100), params=params_hmfast)

   # ------ Plot the results ------
   m_grid = jnp.geomspace(5e10, 3.5e15, 100)
   plt.figure(figsize=(7, 4))
   plt.loglog(m_grid, hmf_z0, label=f"z = 0.0", linestyle='-', color='C0')  # Solid line
   plt.loglog(m_grid, hmf_z1, label=f"z = 1.0", linestyle='--', color='C1') # Dashed line
   plt.loglog(m_grid, hmf_z2, label=f"z = 2.0", linestyle='-.', color='C2') # Dash-dot line
   plt.ylabel(r"$\mathrm{d}n/\mathrm{d}\ln M$")
   plt.ylim([1e-8, 1e-1])
   plt.legend()
   plt.tight_layout()
   plt.show()


.. image:: _static/hmf.png
   :width: 90%
   :align: center
   :alt: Halo mass function


Halo bias function
----------------------------

We can compute the halo bias function in a similar manner.

.. code-block:: python

   hbf_z0 = halo_model.halo_bias_function(0.0, m=jnp.geomspace(5e10, 3.5e15, 100), params=params_hmfast)
   hbf_z1 = halo_model.halo_bias_function(1.0, m=jnp.geomspace(5e10, 3.5e15, 100), params=params_hmfast)
   hbf_z2 = halo_model.halo_bias_function(2.0, m=jnp.geomspace(5e10, 3.5e15, 100), params=params_hmfast)

   # ------ Plot the results ------
   m_grid = jnp.geomspace(5e10, 3.5e15, 100)
   plt.figure(figsize=(7, 4))
   plt.loglog(m_grid, hbf_z0, label=f"z = 0.0", linestyle='-', color='C0')  # Solid line
   plt.loglog(m_grid, hbf_z1, label=f"z = 1.0", linestyle='--', color='C1') # Dashed line
   plt.loglog(m_grid, hbf_z2, label=f"z = 2.0", linestyle='-.', color='C2') # Dash-dot line
   plt.xlabel(r"$M\ [M_\odot/h]$")
   plt.ylabel(r"$b(M)$")
   plt.legend()
   plt.tight_layout()
   plt.show()


.. image:: _static/hbf.png
   :width: 90%
   :align: center
   :alt: Halo bias function


Angular power spectra
---------------------

Once you're ready to compute the angular power spectra, you must first instantiate your tracer object as follows. 

.. code-block:: python

    tsz_tracer = halo_model.create_tracer("y")

You may now easily compute the 1-halo and 2-halo of your tSZ tracer:


.. code-block:: python

   cl_yy_1h = halo_model.cl_1h(tsz_tracer, z=jnp.linspace(0.05, 3.0, 100), m=jnp.geomspace(5e10, 3.5e15, 100), l=jnp.geomspace(2, 8e3, 50), params=params_hmfast)
   cl_yy_2h = halo_model.cl_2h(tsz_tracer, z=jnp.linspace(0.05, 3.0, 100), m=jnp.geomspace(5e10, 3.5e15, 100), l=jnp.geomspace(2, 8e3, 50), params=params_hmfast)

   # ------ Convert to D_l and plot the results ------
   l = jnp.geomspace(2, 8e3, 50)
   dl_yy_1h = l * (l + 1) * cl_yy_1h / (2 * jnp.pi) * 1e12
   dl_yy_2h = l * (l + 1) * cl_yy_2h / (2 * jnp.pi) * 1e12

   plt.figure()
   plt.loglog(l, dl_yy_1h, label="1-halo term")
   plt.loglog(l, dl_yy_2h, label="2-halo term")
   plt.xlabel(r"$\ell$")
   plt.ylabel(r"$10^{12} D_\ell$")
   plt.legend()
   plt.grid(True, which="both", linestyle="--", alpha=0.5)
   plt.show()


.. image:: _static/C_ell_yy.png
   :width: 90%
   :align: center
   :alt: tSZ angular power spectrum

