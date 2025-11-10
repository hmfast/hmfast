"""
Test suite for the EDE-v2 emulator module.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import os
from hmfast.ede_emulator import EDEEmulator

# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


class TestEDEEmulator:
    """Test class for EDEEmulator functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_emulator(self):
        """Set up emulator for testing."""
        # Skip tests if no data path available
        data_path = os.getenv('PATH_TO_CLASS_SZ_DATA')
        if data_path is None:
            pytest.skip("PATH_TO_CLASS_SZ_DATA environment variable not set")
        
        # Check if EDE-v2 emulator files exist
        ede_path = os.path.join(data_path, 'ede-v2')
        if not os.path.exists(ede_path):
            pytest.skip("EDE-v2 emulator data not found")
        
        try:
            self.emulator = EDEEmulator(data_path=data_path)
        except Exception as e:
            pytest.skip(f"Could not initialize emulator: {e}")
    
    def test_default_parameters(self):
        """Test that default parameters are properly set."""
        assert 'fEDE' in EDEEmulator.DEFAULT_PARAMS
        assert 'H0' in EDEEmulator.DEFAULT_PARAMS
        assert EDEEmulator.DEFAULT_PARAMS['fEDE'] == 0.001
        assert EDEEmulator.DEFAULT_PARAMS['H0'] == 67.66
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        valid_params = {
            'fEDE': 0.05,
            'H0': 70.0,
            'omega_b': 0.022,
            'omega_cdm': 0.12
        }
        assert self.emulator.validate_parameters(valid_params)
        
        # Invalid parameters
        invalid_params = {
            'fEDE': 0.5,  # Too high
            'H0': 30.0,   # Too low
        }
        assert not self.emulator.validate_parameters(invalid_params)
    
    def test_angular_distance(self):
        """Test angular distance calculation."""
        params = {
            'fEDE': 0.05,
            'H0': 70.0,
            'omega_b': 0.022,
            'omega_cdm': 0.12,
            'ln10^{10}A_s': 3.047,
            'n_s': 0.965
        }
        
        # Test single redshift
        z = 1.0
        da = self.emulator.get_angular_distance_at_z(z, params)
        assert jnp.isfinite(da)
        assert da > 0
        
        # Test array of redshifts
        z_array = jnp.array([0.1, 0.5, 1.0, 2.0])
        da_array = self.emulator.get_angular_distance_at_z(z_array, params)
        assert da_array.shape == z_array.shape
        assert jnp.all(jnp.isfinite(da_array))
        assert jnp.all(da_array > 0)
        
        # Test that distance decreases with redshift (angular diameter distance)
        assert da_array[0] < da_array[1]  # DA(z) has a maximum
    
    def test_hubble_parameter(self):
        """Test Hubble parameter calculation."""
        params = {
            'fEDE': 0.05,
            'H0': 70.0,
            'omega_b': 0.022,
            'omega_cdm': 0.12
        }
        
        # Test single redshift
        z = 1.0
        hz = self.emulator.get_hubble_at_z(z, params)
        assert jnp.isfinite(hz)
        assert hz > params['H0']  # H(z) > H0 for z > 0
        
        # Test array of redshifts
        z_array = jnp.array([0.0, 0.5, 1.0, 2.0])
        hz_array = self.emulator.get_hubble_at_z(z_array, params)
        assert hz_array.shape == z_array.shape
        assert jnp.all(jnp.isfinite(hz_array))
        
        # Test that H(0) ≈ H0
        assert jnp.abs(hz_array[0] - params['H0']) < 1.0
        
        # Test that Hubble parameter increases with redshift
        assert jnp.all(jnp.diff(hz_array) > 0)
    
    def test_critical_density(self):
        """Test critical density calculation."""
        params = {
            'fEDE': 0.05,
            'H0': 70.0,
            'omega_b': 0.022,
            'omega_cdm': 0.12
        }
        
        # Test single redshift
        z = 1.0
        rho_crit = self.emulator.get_rho_crit_at_z(z, params)
        assert jnp.isfinite(rho_crit)
        assert rho_crit > 0
        
        # Test array of redshifts
        z_array = jnp.array([0.0, 1.0, 2.0])
        rho_array = self.emulator.get_rho_crit_at_z(z_array, params)
        assert rho_array.shape == z_array.shape
        assert jnp.all(jnp.isfinite(rho_array))
        assert jnp.all(rho_array > 0)
        
        # Critical density should increase with redshift
        assert jnp.all(jnp.diff(rho_array) > 0)
    
    def test_linear_power_spectrum(self):
        """Test linear power spectrum calculation."""
        params = {
            'fEDE': 0.05,
            'H0': 70.0,
            'omega_b': 0.022,
            'omega_cdm': 0.12,
            'ln10^{10}A_s': 3.047,
            'n_s': 0.965
        }
        
        # Test at z=0
        z = 0.0
        pk, k = self.emulator.get_pkl_at_z(z, params)
        
        assert len(pk) == len(k)
        assert jnp.all(jnp.isfinite(pk))
        assert jnp.all(jnp.isfinite(k))
        assert jnp.all(pk > 0)
        assert jnp.all(k > 0)
        
        # Test at higher redshift
        z_high = 2.0
        pk_high, k_high = self.emulator.get_pkl_at_z(z_high, params)
        
        # Power should be lower at higher redshift
        assert jnp.all(pk_high < pk)
        
        # Test extrapolation to very high redshift
        z_extrap = 15.0
        pk_extrap, k_extrap = self.emulator.get_pkl_at_z(z_extrap, params)
        assert jnp.all(jnp.isfinite(pk_extrap))
    
    def test_nonlinear_power_spectrum(self):
        """Test nonlinear power spectrum calculation."""
        params = {
            'fEDE': 0.05,
            'H0': 70.0,
            'omega_b': 0.022,
            'omega_cdm': 0.12,
            'ln10^{10}A_s': 3.047,
            'n_s': 0.965
        }
        
        z = 0.0
        pknl, k = self.emulator.get_pknl_at_z(z, params)
        pkl, k_lin = self.emulator.get_pkl_at_z(z, params)
        
        assert len(pknl) == len(k)
        assert jnp.all(jnp.isfinite(pknl))
        assert jnp.all(pknl > 0)
        
        # Nonlinear power should be larger than linear on small scales
        # (assuming k is ordered from small to large)
        if len(pknl) > 100:  # Check if we have enough points
            assert jnp.mean(pknl[-100:]) > jnp.mean(pkl[-100:])
    
    def test_sigma8(self):
        """Test sigma8 calculation.""" 
        params = {
            'fEDE': 0.05,
            'H0': 70.0,
            'omega_b': 0.022,
            'omega_cdm': 0.12,
            'ln10^{10}A_s': 3.047,
            'n_s': 0.965
        }
        
        # Test single redshift
        z = 0.0
        s8 = self.emulator.get_sigma8_at_z(z, params)
        assert jnp.isfinite(s8)
        assert 0.5 < s8 < 1.5  # Reasonable range
        
        # Test array of redshifts
        z_array = jnp.array([0.0, 1.0, 2.0])
        s8_array = self.emulator.get_sigma8_at_z(z_array, params)
        assert s8_array.shape == z_array.shape
        assert jnp.all(jnp.isfinite(s8_array))
        
        # sigma8 should decrease with redshift
        assert jnp.all(jnp.diff(s8_array) < 0)
    
    def test_derived_parameters(self):
        """Test derived parameter calculation."""
        params = {
            'fEDE': 0.05,
            'H0': 70.0,
            'omega_b': 0.022,
            'omega_cdm': 0.12,
            'ln10^{10}A_s': 3.047,
            'n_s': 0.965
        }
        
        derived = self.emulator.get_derived_parameters(params)
        
        # Check that expected parameters are present
        expected_params = ['sigma8', '100*theta_s', 'h', 'Omega_m']
        for param in expected_params:
            assert param in derived
            assert jnp.isfinite(derived[param])
        
        # Check some physical constraints
        assert derived['h'] > 0.5
        assert derived['h'] < 1.0
        assert derived['Omega_m'] > 0.1
        assert derived['Omega_m'] < 0.6
        assert derived['sigma8'] > 0.5
        assert derived['sigma8'] < 1.5
    
    def test_cmb_spectra(self):
        """Test CMB spectra calculation."""
        params = {
            'fEDE': 0.05,
            'H0': 70.0,
            'omega_b': 0.022,
            'omega_cdm': 0.12,
            'ln10^{10}A_s': 3.047,
            'n_s': 0.965,
            'tau_reio': 0.06
        }
        
        spectra = self.emulator.get_cmb_spectra(params, lmax=2000)
        
        # Check that all expected spectra are present
        expected_spectra = ['ell', 'tt', 'te', 'ee', 'pp']
        for spec in expected_spectra:
            assert spec in spectra
            assert len(spectra[spec]) > 0
            assert jnp.all(jnp.isfinite(spectra[spec]))
        
        # Check that all arrays have the same length
        n_ell = len(spectra['ell'])
        for spec in ['tt', 'te', 'ee', 'pp']:
            assert len(spectra[spec]) == n_ell
        
        # Check that ell starts at 2
        assert spectra['ell'][0] == 2
        
        # Check that TT spectrum is positive
        assert jnp.all(spectra['tt'] > 0)
    
    def test_jax_compatibility(self):
        """Test JAX JIT compilation and differentiation."""
        params = {
            'fEDE': 0.05,
            'H0': 70.0,
            'omega_b': 0.022,
            'omega_cdm': 0.12,
            'ln10^{10}A_s': 3.047,
            'n_s': 0.965
        }
        
        # Test JIT compilation
        @jax.jit
        def test_function(z_val):
            return self.emulator.get_angular_distance_at_z(z_val, params)
        
        z = 1.0
        result1 = test_function(z)
        result2 = test_function(z)  # Should use compiled version
        
        assert jnp.allclose(result1, result2)
        
        # Test gradient calculation
        def angular_distance_wrapper(h0):
            test_params = params.copy()
            test_params['H0'] = h0
            return self.emulator.get_angular_distance_at_z(1.0, test_params)
        
        # Calculate gradient
        grad_func = jax.grad(angular_distance_wrapper)
        gradient = grad_func(70.0)
        
        assert jnp.isfinite(gradient)
        assert jnp.abs(gradient) > 0  # Should have non-zero sensitivity to H0
    
    def test_ede_parameters(self):
        """Test EDE-specific parameters."""
        # Test with no EDE (fEDE = 0)
        params_no_ede = {
            'fEDE': 0.0,
            'H0': 70.0,
            'omega_b': 0.022,
            'omega_cdm': 0.12,
            'ln10^{10}A_s': 3.047,
            'n_s': 0.965
        }
        
        # Test with EDE
        params_with_ede = params_no_ede.copy()
        params_with_ede['fEDE'] = 0.1
        params_with_ede['log10z_c'] = 3.5
        params_with_ede['thetai_scf'] = 2.8
        
        # Both should produce valid results
        da_no_ede = self.emulator.get_angular_distance_at_z(1.0, params_no_ede)
        da_with_ede = self.emulator.get_angular_distance_at_z(1.0, params_with_ede)
        
        assert jnp.isfinite(da_no_ede)
        assert jnp.isfinite(da_with_ede)
        
        # EDE should affect distances
        assert jnp.abs(da_no_ede - da_with_ede) > 0.01  # Should have measurable effect


@pytest.mark.benchmark
class TestEDEEmulatorPerformance:
    """Performance benchmarks for the emulator."""
    
    def test_timing(self, benchmark):
        """Benchmark emulator speed.""" 
        data_path = os.getenv('PATH_TO_CLASS_SZ_DATA')
        if data_path is None:
            pytest.skip("PATH_TO_CLASS_SZ_DATA environment variable not set")
        
        emulator = EDEEmulator(data_path=data_path)
        
        params = {
            'fEDE': 0.05,
            'H0': 70.0,
            'omega_b': 0.022,
            'omega_cdm': 0.12,
            'ln10^{10}A_s': 3.047,
            'n_s': 0.965
        }
        
        # Benchmark angular distance calculation
        result = benchmark(emulator.get_angular_distance_at_z, 1.0, params)
        assert jnp.isfinite(result)


if __name__ == "__main__":
    # Run basic tests if called directly
    pytest.main([__file__, "-v"])