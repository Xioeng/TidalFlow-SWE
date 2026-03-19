"""Tests for configuration module."""

import json
from pathlib import Path

import pytest

from swe_simulator.config import SimulationConfig


class TestSimulationConfigValidation:
    """Test configuration validation."""

    def test_valid_config_creation(self, basic_config):
        """Test that valid config is created without errors."""
        assert basic_config.nx == 50
        assert basic_config.ny == 50
        assert basic_config.lon_range == (-1.0, 1.0)
        assert basic_config.lat_range == (-1.0, 1.0)

    def test_invalid_lon_range_missing(self):
        """Test error when lon_range is missing."""
        with pytest.raises(ValueError):
            SimulationConfig(
                lat_range=(-1.0, 1.0),
                nx=50,
                ny=50,
            )

    def test_invalid_lat_range_missing(self):
        """Test error when lat_range is missing."""
        with pytest.raises(ValueError):
            SimulationConfig(
                lon_range=(-1.0, 1.0),
                nx=50,
                ny=50,
            )

    def test_invalid_lon_range_reversed(self):
        """Test error when lon_min > lon_max."""
        with pytest.raises(ValueError):
            SimulationConfig(
                lon_range=(1.0, -1.0),  # reversed
                lat_range=(-1.0, 1.0),
                nx=50,
                ny=50,
            )

    def test_invalid_lat_range_reversed(self):
        """Test error when lat_min > lat_max."""
        with pytest.raises(ValueError):
            SimulationConfig(
                lon_range=(-1.0, 1.0),
                lat_range=(1.0, -1.0),  # reversed
                nx=50,
                ny=50,
            )

    def test_invalid_nx_zero(self):
        """Test error when nx is zero."""
        with pytest.raises(ValueError):
            SimulationConfig(
                lon_range=(-1.0, 1.0),
                lat_range=(-1.0, 1.0),
                nx=0,
                ny=50,
            )

    def test_invalid_ny_negative(self):
        """Test error when ny is negative."""
        with pytest.raises(ValueError):
            SimulationConfig(
                lon_range=(-1.0, 1.0),
                lat_range=(-1.0, 1.0),
                nx=50,
                ny=-10,
            )

    def test_config_default_values(self):
        """Test that defaults are applied correctly."""
        config = SimulationConfig(
            lon_range=(-1.0, 1.0),
            lat_range=(-1.0, 1.0),
        )
        assert config.gravity == 9.81
        assert config.cfl_desired == 0.9
        assert config.cfl_max == 1.0

    def test_config_custom_gravity(self):
        """Test custom gravity value."""
        config = SimulationConfig(
            lon_range=(-1.0, 1.0),
            lat_range=(-1.0, 1.0),
            gravity=9.8,
        )
        assert config.gravity == 9.8

    def test_config_to_dict(self, basic_config):
        """Test conversion to dictionary."""
        config_dict = basic_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["nx"] == 50
        assert config_dict["gravity"] == 9.81

    def test_config_from_json(self, tmp_path):
        """Test loading config from JSON file."""
        # Create a JSON config file
        config_file = tmp_path / "config.json"
        config_dict = {
            "lon_range": [-1.0, 1.0],
            "lat_range": [-1.0, 1.0],
            "nx": 50,
            "ny": 50,
            "t_final": 10.0,
            "dt": 0.1,
            "gravity": 9.81,
        }
        with open(config_file, "w") as f:
            json.dump(config_dict, f)

        # Load it back
        config = SimulationConfig.load(str(config_file))
        assert config.nx == 50
        assert config.gravity == 9.81
