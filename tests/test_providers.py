"""Tests for data providers."""

import numpy as np
import pytest

from swe_simulator.config import SimulationConfig
from swe_simulator.providers import (
    BathymetryProvider,
    InitialConditionProvider,
    WindProvider,
)
from swe_simulator.providers_examples import (
    ConstantWind,
    FlatBathymetry,
    GaussianHumpInitialCondition,
    SlopingBathymetry,
    TimeVaryingWind,
)


class TestInitialConditionProvider:
    """Test initial condition provider interface and implementations."""

    def test_abstract_class_cannot_be_instantiated(self):
        """InitialConditionProvider should not be instantiable."""
        with pytest.raises(TypeError):
            InitialConditionProvider()

    def test_gaussian_hump_shape(self, basic_config):
        """Gaussian hump should return correct shape."""
        ic = GaussianHumpInitialCondition()
        result = ic.get_initial_condition(basic_config)

        assert result.shape == (3, basic_config.ny, basic_config.nx)

    def test_gaussian_hump_values(self, basic_config):
        """Gaussian hump values should be in reasonable ranges."""
        ic = GaussianHumpInitialCondition(height=2.0)
        result = ic.get_initial_condition(basic_config)

        h, hu, hv = result[0], result[1], result[2]

        # Height should be positive
        assert np.all(h >= 0)
        # Max height should be close to specified height
        assert np.isclose(np.max(h), 2.0, rtol=0.01)
        # Momentum should be zero (no initial flow)
        assert np.allclose(hu, 0.0)
        assert np.allclose(hv, 0.0)


class TestWindProvider:
    """Test wind provider interface and implementations."""

    def test_abstract_class_cannot_be_instantiated(self):
        """WindProvider should not be instantiable."""
        with pytest.raises(TypeError):
            WindProvider()

    def test_constant_wind(self):
        """Constant wind should return same values."""
        wind = ConstantWind(u_wind=5.0, v_wind=2.0)

        u1, v1 = wind.get_wind(time=0.0)
        u2, v2 = wind.get_wind(time=10.0)

        assert u1 == 5.0 and v1 == 2.0
        assert u2 == 5.0 and v2 == 2.0

    def test_time_varying_wind(self):
        """Time-varying wind should change with time."""
        # Linear functions
        u_func = lambda t: 2.0 * t
        v_func = lambda t: -t

        wind = TimeVaryingWind(u_func, v_func)

        u0, v0 = wind.get_wind(time=0.0)
        u1, v1 = wind.get_wind(time=5.0)

        assert u0 == 0.0 and v0 == 0.0
        assert u1 == 10.0 and v1 == -5.0


class TestBathymetryProvider:
    """Test bathymetry provider interface and implementations."""

    def test_abstract_class_cannot_be_instantiated(self):
        """BathymetryProvider should not be instantiable."""
        with pytest.raises(TypeError):
            BathymetryProvider()

    def test_flat_bathymetry_shape(self, basic_config):
        """Flat bathymetry should return correct shape."""
        bathy = FlatBathymetry(depth=-10.0)
        result = bathy.get_bathymetry(basic_config)

        assert result.shape == (basic_config.ny, basic_config.nx)

    def test_flat_bathymetry_values(self, basic_config):
        """Flat bathymetry should have uniform depth."""
        depth = -15.0
        bathy = FlatBathymetry(depth=depth)
        result = bathy.get_bathymetry(basic_config)

        assert np.allclose(result, depth)

    def test_sloping_bathymetry(self, basic_config):
        """Sloping bathymetry should vary with y-coordinate."""
        bathy = SlopingBathymetry(depth_min=-5.0, depth_max=-20.0)
        result = bathy.get_bathymetry(basic_config)

        assert result.shape == (basic_config.ny, basic_config.nx)
        # Depth should increase from min to max
        assert np.min(result) >= -20.0
        assert np.max(result) <= -5.0
