"""
EDE-v2 cosmological emulator for hmfast.

This module provides fast JAX-compatible emulated cosmological calculations
specifically for the EDE-v2 (Early Dark Energy version 2) model, removing
the dependency on classy_szfast while maintaining the same interface.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Union
from functools import partial

from .clean_nn_emulator import CleanRestoreNN as RestoreNN, CleanRestorePCAplusNN as RestorePCAplusNN

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


class EDEEmulator:
    """
    EDE-v2 cosmological emulator with JAX compatibility.
    
    Provides fast emulated predictions for cosmological quantities
    using the EDE-v2 (Early Dark Energy v2) model emulators.
    """
    
    # EDE-v2 emulator metadata
    EMULATOR_DICT = {
        'TT': 'TT_v2',
        'TE': 'TE_v2', 
        'EE': 'EE_v2',
        'PP': 'PP_v2',
        'PKNL': 'PKNL_v2',
        'PKL': 'PKL_v2',
        'DER': 'DER_v2',
        'DAZ': 'DAZ_v2',
        'HZ': 'HZ_v2',
        'S8Z': 'S8Z_v2'
    }
    
    # Default EDE-v2 parameters
    DEFAULT_PARAMS = {
        'fEDE': 0.001,
        'tau_reio': 0.054,
        'H0': 67.66,
        'ln10^{10}A_s': 3.047,
        'omega_b': 0.02242,
        'omega_cdm': 0.11933,
        'n_s': 0.9665,
        'log10z_c': 3.562,
        'thetai_scf': 2.83,
        'r': 0.,
        'N_ur': 0.00441,  # For Neff = 3.044
        'N_ncdm': 1,
        'deg_ncdm': 3,
        'm_ncdm': 0.02,
        'T_cmb': 2.7255
    }
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the EDE-v2 emulator.
        
        Parameters
        ----------
        data_path : str, optional
            Path to emulator data directory. If None, uses environment variable.
        """
        # Get data path using same logic as classy_szfast
        self.data_path = self._get_data_path(data_path)
        self.emulator_path = os.path.join(self.data_path, 'ede_v2_numpy')
        
        # Initialize emulator storage
        self._emulators = {}
        self._load_emulators()
        
        # Set up interpolation grids using actual emulator modes
        self._setup_interpolation_grids_post_load()
    
    def _get_data_path(self, provided_path: Optional[str] = None) -> str:
        """
        Get the data path using the same logic as classy_szfast.
        
        Parameters
        ----------
        provided_path : str, optional
            User-provided path
            
        Returns
        -------
        str
            Path to data directory
        """
        if provided_path is not None:
            if os.path.exists(provided_path):
                return provided_path
            else:
                raise ValueError(f"Provided data path does not exist: {provided_path}")
        
        # Check environment variable (same as classy_szfast)
        env_path = os.getenv('PATH_TO_CLASS_SZ_DATA')
        
        if env_path is not None:
            # Handle case where env var might end with class_sz_data_directory
            if env_path.endswith("class_sz_data_directory"):
                base_path = env_path
            else:
                base_path = os.path.join(env_path, "class_sz_data_directory")
                
            if os.path.exists(base_path):
                return base_path
            else:
                print(f"Warning: PATH_TO_CLASS_SZ_DATA points to non-existent directory: {base_path}")
        
        # Fall back to default location (same as get_cosmopower_emus)
        home_dir = os.path.expanduser("~")
        default_path = os.path.join(home_dir, "class_sz_data_directory")
        
        if os.path.exists(default_path):
            return default_path
        
        # Last resort: check common locations
        common_paths = [
            "/usr/local/share/class_sz_data",
            os.path.join(os.getcwd(), "..", "class_sz_data"),
            os.path.join(os.getcwd(), "class_sz_data"),
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'ede-v2')):
                return path
        
        raise ValueError(
            f"Could not find emulator data directory. Tried:\n"
            f"  - Provided path: {provided_path}\n"
            f"  - Environment variable PATH_TO_CLASS_SZ_DATA: {env_path}\n"
            f"  - Default: {default_path}\n"
            f"  - Common locations: {common_paths}\n"
            f"Please set PATH_TO_CLASS_SZ_DATA or provide data_path explicitly."
        )
    
    def _load_emulators(self):
        """Load all EDE-v2 emulators."""
        if not os.path.exists(self.emulator_path):
            self._print_directory_info()
            raise FileNotFoundError(
                f"EDE emulator directory not found: {self.emulator_path}\n"
                f"Please ensure this directory exists in your data path."
            )
        
        try:
            # Load CMB emulators
            self._emulators['TT'] = RestoreNN(
                os.path.join(self.emulator_path, 'TTTEEE', self.EMULATOR_DICT['TT'])
            )
            self._emulators['TE'] = RestorePCAplusNN(  # TE uses PCA compression
                os.path.join(self.emulator_path, 'TTTEEE', self.EMULATOR_DICT['TE'])
            )
            self._emulators['EE'] = RestoreNN(
                os.path.join(self.emulator_path, 'TTTEEE', self.EMULATOR_DICT['EE'])
            )
            
            # Load lensing emulator
            self._emulators['PP'] = RestoreNN(
                os.path.join(self.emulator_path, 'PP', self.EMULATOR_DICT['PP'])
            )
            
            # Load power spectrum emulators
            self._emulators['PKNL'] = RestoreNN(
                os.path.join(self.emulator_path, 'PK', self.EMULATOR_DICT['PKNL'])
            )
            self._emulators['PKL'] = RestoreNN(
                os.path.join(self.emulator_path, 'PK', self.EMULATOR_DICT['PKL'])
            )
            
            # Load derived parameter emulator
            self._emulators['DER'] = RestoreNN(
                os.path.join(self.emulator_path, 'derived-parameters', self.EMULATOR_DICT['DER'])
            )
            
            # Load distance and growth emulators
            self._emulators['DAZ'] = RestoreNN(
                os.path.join(self.emulator_path, 'growth-and-distances', self.EMULATOR_DICT['DAZ'])
            )
            self._emulators['HZ'] = RestoreNN(
                os.path.join(self.emulator_path, 'growth-and-distances', self.EMULATOR_DICT['HZ'])
            )
            self._emulators['S8Z'] = RestoreNN(
                os.path.join(self.emulator_path, 'growth-and-distances', self.EMULATOR_DICT['S8Z'])
            )
            
            print(f"✓ Successfully loaded {len(self._emulators)} EDE emulators from {self.emulator_path}")
            
        except Exception as e:
            self._print_directory_info()
            raise RuntimeError(f"Failed to load EDE emulators: {e}")
    
    def _print_directory_info(self):
        """Print helpful information about the data directory structure."""
        print(f"\nDEBUG: Data path information:")
        print(f"  Base data path: {self.data_path}")
        print(f"  EDE-v2 path: {self.emulator_path}")
        print(f"  Base path exists: {os.path.exists(self.data_path)}")
        print(f"  EDE-v2 path exists: {os.path.exists(self.emulator_path)}")
        
        if os.path.exists(self.data_path):
            print(f"  Contents of {self.data_path}:")
            try:
                for item in sorted(os.listdir(self.data_path)):
                    item_path = os.path.join(self.data_path, item)
                    item_type = "dir" if os.path.isdir(item_path) else "file"
                    print(f"    - {item} ({item_type})")
            except PermissionError:
                print("    [Permission denied]")
        
        if os.path.exists(self.emulator_path):
            print(f"  Contents of {self.emulator_path}:")
            try:
                for item in sorted(os.listdir(self.emulator_path)):
                    item_path = os.path.join(self.emulator_path, item)
                    item_type = "dir" if os.path.isdir(item_path) else "file"
                    print(f"    - {item} ({item_type})")
            except PermissionError:
                print("    [Permission denied]")
        
        print(f"\n  Expected structure:")
        print(f"  {self.emulator_path}/")
        for subdir, files in [
            ('TTTEEE', ['TT_v2.npz', 'TE_v2.npz', 'EE_v2.npz']),
            ('PP', ['PP_v2.npz']),
            ('PK', ['PKNL_v2.npz', 'PKL_v2.npz']),
            ('derived-parameters', ['DER_v2.npz']),
            ('growth-and-distances', ['DAZ_v2.npz', 'HZ_v2.npz', 'S8Z_v2.npz'])
        ]:
            print(f"    ├── {subdir}/")
            for file in files:
                print(f"    │   └── {file}")
        print()
    
    def _setup_interpolation_grids(self):
        """Set up interpolation grids for z-dependent quantities."""
        # Get actual z-grids from emulators after loading
        pass  # Will be called after _load_emulators()
        
    def _setup_interpolation_grids_post_load(self):
        """Set up interpolation grids after emulators are loaded."""
        # Get z-grid from DAZ emulator (z=1 to z=4999)
        self.daz_z_grid = jnp.array(self._emulators['DAZ'].modes, dtype=jnp.float64)
        
        # Get z-grid from HZ emulator
        self.hz_z_grid = jnp.array(self._emulators['HZ'].modes, dtype=jnp.float64)
        
        # Get z-grid from S8Z emulator  
        self.s8z_z_grid = jnp.array(self._emulators['S8Z'].modes, dtype=jnp.float64)
        
        # k grid for power spectra (get from PKL emulator)
        self.k_grid = jnp.array(self._emulators['PKL'].modes, dtype=jnp.float64)
        
        # Maximum redshift for power spectrum grid
        self.pk_grid_zmax = 4999.0
    
    def _merge_with_defaults(self, params_dict: Dict[str, Union[float, jnp.ndarray]]) -> Dict[str, Union[float, jnp.ndarray]]:
        """Merge user parameters with defaults."""
        merged = self.DEFAULT_PARAMS.copy()
        merged.update(params_dict)
        return merged
    
    @partial(jax.jit, static_argnums=(0,))
    def _interpolate_z_dependent(self, 
                                z_requested: Union[float, jnp.ndarray], 
                                predictions: jnp.ndarray, 
                                z_grid: jnp.ndarray) -> jnp.ndarray:
        """
        Interpolate z-dependent quantities to requested redshifts.
        
        Parameters
        ----------
        z_requested : float or jnp.ndarray
            Requested redshift(s)
        predictions : jnp.ndarray
            Emulator predictions on z_grid
        z_grid : jnp.ndarray
            Redshift grid used for predictions
            
        Returns
        -------
        jnp.ndarray
            Interpolated values
        """
        # Ensure z_requested is an array
        z_req = jnp.atleast_1d(z_requested)
        
        # Perform linear interpolation
        result = jnp.interp(z_req, z_grid, predictions, left=jnp.nan, right=jnp.nan)
        
        # Return scalar if input was scalar
        if jnp.ndim(z_requested) == 0:
            return result[0]
        return result
    
    def get_angular_distance_at_z(self, 
                                 z: Union[float, jnp.ndarray], 
                                 params_values_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """
        Get angular diameter distance at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Angular diameter distance(s) in Mpc
        """
        merged_params = self._merge_with_defaults(params_values_dict)
        
        # Get predictions on full grid
        da_predictions = self._emulators['DAZ'].predictions(merged_params)
        
        # Interpolate to requested redshifts
        return self._interpolate_z_dependent(z, da_predictions, self.daz_z_grid) / (1.0 + z)
    
    def get_hubble_at_z(self, 
                       z: Union[float, jnp.ndarray], 
                       params_values_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """
        Get Hubble parameter at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Hubble parameter(s) in km/s/Mpc
        """
        merged_params = self._merge_with_defaults(params_values_dict)
        
        # Get predictions on full grid
        hz_predictions = self._emulators['HZ'].predictions(merged_params)
        
        # Interpolate to requested redshifts
        return self._interpolate_z_dependent(z, hz_predictions, self.hz_z_grid)
    
    def get_rho_crit_at_z(self, 
                         z: Union[float, jnp.ndarray], 
                         params_values_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """
        Get critical density at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Critical density(s) in (Msun/h) / (Mpc/h)^3
        """
        # Get Hubble parameter
        H_z = self.get_hubble_at_z(z, params_values_dict)
        
        # Convert to critical density
        # rho_crit = 3 H^2 / (8 pi G)
        # Using units: rho_crit_over_h2_in_GeV_per_cm3 = 1.0537e-5
        rho_crit_factor = 2.77536627e11  # (Msun/h) / (Mpc/h)^3 / (km/s/Mpc)^2
        
        return rho_crit_factor * (H_z / 100.0)**2
    
    def get_pkl_at_z(self, 
                    z: Union[float, jnp.ndarray], 
                    params_values_dict: Dict[str, Union[float, jnp.ndarray]]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get linear power spectrum at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict
            Cosmological parameters
            
        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Linear power spectrum and k array
        """
        merged_params = self._merge_with_defaults(params_values_dict)
        
        # Add redshift to parameters for prediction
        z_val = jnp.atleast_1d(z)[0] if hasattr(z, '__len__') else z
        merged_params['z_pk_save_nonclass'] = float(z_val)
        
        # Handle extrapolation beyond training redshift
        if z_val > self.pk_grid_zmax:
            # Get power spectrum at maximum redshift and scale
            z_max = self.pk_grid_zmax
            merged_params_zmax = merged_params.copy()
            merged_params_zmax['z_pk_save_nonclass'] = float(z_max)
            
            # Get power spectrum at z_max
            pkl_zmax = self._emulators['PKL'].predictions(merged_params_zmax)
            
            # Scale by growth factor: P(k,z) ~ D(z)^2
            # In matter domination: D(z) ~ 1/(1+z)
            scale_factor = ((1 + z_max) / (1 + z_val))**2
            pkl = pkl_zmax * scale_factor
        else:
            # Direct prediction
            pkl = self._emulators['PKL'].predictions(merged_params)
        
        return pkl, self.k_grid
    
    def get_pknl_at_z(self, 
                     z: Union[float, jnp.ndarray], 
                     params_values_dict: Dict[str, Union[float, jnp.ndarray]]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get nonlinear power spectrum at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict
            Cosmological parameters
            
        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Nonlinear power spectrum and k array
        """
        merged_params = self._merge_with_defaults(params_values_dict)
        
        z_val = jnp.atleast_1d(z)[0] if hasattr(z, '__len__') else z
        merged_params['z_pk_save_nonclass'] = float(z_val)
        
        # Handle extrapolation beyond training redshift
        if z_val > self.pk_grid_zmax:
            # Use linear power spectrum scaled by growth factor
            pkl, k = self.get_pkl_at_z(z, params_values_dict)
            # For high z, nonlinear and linear should be similar
            return pkl, k
        else:
            # Direct prediction
            pknl = self._emulators['PKNL'].predictions(merged_params)
            return pknl, self.k_grid
    
    def get_sigma8_at_z(self, 
                       z: Union[float, jnp.ndarray], 
                       params_values_dict: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """
        Get sigma8 at redshift z.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift(s)
        params_values_dict : dict
            Cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            sigma8 value(s)
        """
        merged_params = self._merge_with_defaults(params_values_dict)
        
        # Get predictions on full grid
        s8_predictions = self._emulators['S8Z'].predictions(merged_params)
        
        # Interpolate to requested redshifts
        return self._interpolate_z_dependent(z, s8_predictions, self.s8z_z_grid)
    
    def get_derived_parameters(self, 
                             params_values_dict: Dict[str, Union[float, jnp.ndarray]]) -> Dict[str, jnp.ndarray]:
        """
        Get derived cosmological parameters.
        
        Parameters
        ----------
        params_values_dict : dict
            Input cosmological parameters
            
        Returns
        -------
        dict
            Dictionary of derived parameters
        """
        merged_params = self._merge_with_defaults(params_values_dict)
        
        # Get derived parameter predictions
        derived_pred = self._emulators['DER'].predictions(merged_params)
        
        # Parameter names (from classy_szfast emulators_meta_data.py)
        param_names = [
            '100*theta_s',
            'sigma8', 
            'YHe',
            'z_reio',
            'Neff',
            'tau_rec',
            'z_rec',
            'rs_rec',
            'ra_rec',
            'tau_star',
            'z_star',
            'rs_star',
            'ra_star',
            'rs_drag'
        ]
        
        # Create dictionary
        derived_dict = {}
        for i, name in enumerate(param_names):
            if i < len(derived_pred):
                derived_dict[name] = derived_pred[i]
        
        # Add h and Omega_m from input parameters
        if 'H0' in merged_params:
            derived_dict['h'] = merged_params['H0'] / 100.0
        if 'omega_b' in merged_params and 'omega_cdm' in merged_params and 'H0' in merged_params:
            h = merged_params['H0'] / 100.0
            derived_dict['Omega_m'] = (merged_params['omega_b'] + merged_params['omega_cdm']) / h**2
        
        return derived_dict
    
    def get_cmb_cls(self, 
                    params_values_dict: Dict[str, Union[float, jnp.ndarray]], 
                    lmax: int = 2500) -> Dict[str, jnp.ndarray]:
        """
        Get raw CMB power spectra C_ℓ values (without ℓ(ℓ+1)/(2π) factor).
        
        This method returns the raw C_ℓ power spectra. To plot these values,
        you would need to multiply by ℓ(ℓ+1)/(2π) to get D_ℓ.
        
        Parameters
        ----------
        params_values_dict : dict
            Cosmological parameters
        lmax : int
            Maximum multipole
            
        Returns
        -------
        dict
            Dictionary with 'ell', 'tt', 'te', 'ee', 'pp' arrays containing raw C_ℓ values.
            For plotting, multiply by ℓ(ℓ+1)/(2π) or use get_cmb_spectra() instead.
        """
        # Get what are effectively D_ℓ values
        dls = self._get_raw_cmb_spectra(params_values_dict, lmax)
        
        # Convert D_ℓ → C_ℓ by dividing by ℓ(ℓ+1)/(2π)
        ell = dls['ell']
        dl_to_cl_factor = (2.0 * jnp.pi) / (ell * (ell + 1.0))
        
        spectra = {
            'ell': ell,
            'tt': dls['tt'] * dl_to_cl_factor,
            'te': dls['te'] * dl_to_cl_factor,
            'ee': dls['ee'] * dl_to_cl_factor, 
            # PP: convert from ℓ²(ℓ+1)²C_ℓ/(2π) to C_ℓ
            'pp': dls['pp'] * dl_to_cl_factor / (ell * (ell + 1.0))
        }
        
        return spectra
    
    def get_cmb_spectra(self, 
                       params_values_dict: Dict[str, Union[float, jnp.ndarray]], 
                       lmax: int = 2500) -> Dict[str, jnp.ndarray]:
        """
        Get CMB power spectra D_ℓ for plotting (NO additional factor needed).
        
        This method returns D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) values ready for direct plotting.
        The returned values already include the ℓ(ℓ+1)/(2π) factor.
        
        Parameters
        ----------
        params_values_dict : dict
            Cosmological parameters
        lmax : int
            Maximum multipole
            
        Returns
        -------
        dict
            Dictionary with 'ell', 'tt', 'te', 'ee', 'pp' arrays containing D_ℓ values.
            These values can be plotted directly without multiplying by ℓ(ℓ+1)/(2π).
        """
        return self._get_raw_cmb_spectra(params_values_dict, lmax)
    
    def _get_raw_cmb_spectra(self, 
                            params_values_dict: Dict[str, Union[float, jnp.ndarray]], 
                            lmax: int = 2500) -> Dict[str, jnp.ndarray]:
        """
        Get raw CMB spectra from emulators (these are effectively D_ℓ values).
        
        Parameters
        ----------
        params_values_dict : dict
            Cosmological parameters
        lmax : int
            Maximum multipole
            
        Returns
        -------
        dict
            Dictionary with 'ell', 'tt', 'te', 'ee', 'pp' arrays
        """
        merged_params = self._merge_with_defaults(params_values_dict)
        
        # Get predictions using ten_to_predictions for TT, EE, PP (log-scaled)
        tt_pred = self._emulators['TT'].ten_to_predictions(merged_params)
        ee_pred = self._emulators['EE'].ten_to_predictions(merged_params)  
        pp_pred = self._emulators['PP'].ten_to_predictions(merged_params)
        
        # TE uses regular predictions (not log-scaled)
        te_pred = self._emulators['TE'].predictions(merged_params)
        
        # CMB temperature in microK
        Tcmb_uk = 2.7255e6  # microK
        
        # Create ell array starting from ell=2
        n_tt = len(tt_pred)
        n_te = len(te_pred) 
        n_ee = len(ee_pred)
        n_pp = len(pp_pred)
        
        n_max = min(n_tt, n_te, n_ee, n_pp, lmax - 1)  # -1 because ell starts from 2
        ell = jnp.arange(2, n_max + 2)
        
        # Emulators output D_ℓ/T_CMB² values, so T_CMB² scaling gives D_ℓ
        # These D_ℓ values can be plotted directly (no need to multiply by ℓ(ℓ+1)/(2π))
        spectra = {
            'ell': ell,
            'tt': (Tcmb_uk**2) * tt_pred[:n_max], 
            'te': (Tcmb_uk**2) * te_pred[:n_max],
            'ee': (Tcmb_uk**2) * ee_pred[:n_max],
            'pp': pp_pred[:n_max] / 4.0  # PP scaling as in classy_szfast
        }
        
        return spectra
    
    def validate_parameters(self, params_dict: Dict[str, Union[float, jnp.ndarray]]) -> bool:
        """
        Validate that parameters are within emulator training ranges.
        
        Parameters
        ----------
        params_dict : dict
            Parameters to validate
            
        Returns
        -------
        bool
            True if parameters are valid
        """
        # Define valid ranges for EDE-v2 parameters
        valid_ranges = {
            'fEDE': (0.0, 0.15),
            'H0': (50.0, 90.0),
            'omega_b': (0.015, 0.035),
            'omega_cdm': (0.08, 0.2),
            'ln10^{10}A_s': (2.5, 3.5),
            'n_s': (0.85, 1.15),
            'tau_reio': (0.02, 0.15),
            'log10z_c': (3.0, 4.5),
            'thetai_scf': (1.0, 5.0),
        }
        
        for param, value in params_dict.items():
            if param in valid_ranges:
                min_val, max_val = valid_ranges[param]
                val = float(value)
                if not (min_val <= val <= max_val):
                    return False
        
        return True