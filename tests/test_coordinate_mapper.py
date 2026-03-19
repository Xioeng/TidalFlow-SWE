"""Tests for coordinate mapper module."""

import numpy as np
import pytest

from swe_simulator.coordinate_mapper import GeographicCoordinateMapper


class TestGeographicCoordinateMapper:
    """Test coordinate transformations."""

    def test_mapper_initialization(self, coordinate_mapper):
        """Test mapper initializes with correct center."""
        assert coordinate_mapper.lon0 == 0.0
        assert coordinate_mapper.lat0 == 0.0
        assert coordinate_mapper.R > 0  # Earth radius should be positive

    def test_mapper_custom_center(self):
        """Test mapper with custom center coordinates."""
        mapper = GeographicCoordinateMapper(lon0=-80.5, lat0=25.5)
        assert mapper.lon0 == -80.5
        assert mapper.lat0 == 25.5

    def test_coord_to_metric_single_point(self, coordinate_mapper):
        """Test conversion of single point."""
        x_m, y_m = coordinate_mapper.coord_to_metric(0.0, 0.0)
        # At the origin, metric coords should be close to (0, 0)
        assert np.isclose(x_m, 0.0, atol=1e-6)
        assert np.isclose(y_m, 0.0, atol=1e-6)

    def test_coord_to_metric_offset(self, coordinate_mapper):
        """Test conversion with offset from center."""
        # Small offset from center
        x_m, y_m = coordinate_mapper.coord_to_metric(0.01, 0.01)
        # Should be non-zero
        assert x_m != 0.0
        assert y_m != 0.0

    def test_coord_to_metric_array(self, coordinate_mapper):
        """Test conversion of coordinate arrays."""
        lons = np.array([0.0, 0.1, -0.1])
        lats = np.array([0.0, 0.1, -0.1])
        x_m, y_m = coordinate_mapper.coord_to_metric(lons, lats)

        assert x_m.shape == lons.shape
        assert y_m.shape == lats.shape
        assert np.isclose(x_m[0], 0.0, atol=1e-6)
        assert np.isclose(y_m[0], 0.0, atol=1e-6)

    def test_coord_to_metric_meshgrid(self, coordinate_mapper):
        """Test conversion of meshgrid coordinates."""
        lon = np.linspace(-0.1, 0.1, 11)
        print(lon)
        lat = np.linspace(-0.1, 0.1, 11)
        LON, LAT = np.meshgrid(lon, lat)

        X, Y = coordinate_mapper.coord_to_metric(LON, LAT)

        assert X.shape == LON.shape
        assert Y.shape == LAT.shape
        # Center should be near origin
        assert np.isclose(X[5, 5], 0.0, atol=1)
        assert np.isclose(Y[5, 5], 0.0, atol=1)

    def test_metric_to_coord_single_point(self, coordinate_mapper):
        """Test conversion from metric back to geographic."""
        # Start with geographic
        lon_orig, lat_orig = 0.0, 0.0
        x_m, y_m = coordinate_mapper.coord_to_metric(lon_orig, lat_orig)
        # Convert back
        lon_back, lat_back = coordinate_mapper.metric_to_coord(x_m, y_m)

        assert np.isclose(lon_back, lon_orig, atol=1e-6)
        assert np.isclose(lat_back, lat_orig, atol=1e-6)

    def test_round_trip_conversion(self, coordinate_mapper):
        """Test round-trip conversion (geo → metric → geo)."""
        lons = np.array([-0.05, 0.0, 0.05])
        lats = np.array([-0.05, 0.0, 0.05])

        # Forward transformation
        x_m, y_m = coordinate_mapper.coord_to_metric(lons, lats)
        # Inverse transformation
        lon_back, lat_back = coordinate_mapper.metric_to_coord(x_m, y_m)

        # Should match original (within numerical precision)
        np.testing.assert_allclose(lon_back, lons, rtol=1e-5)
        np.testing.assert_allclose(lat_back, lats, rtol=1e-5)

    def test_coordinate_distance(self, coordinate_mapper):
        """Test distance calculations."""
        # Distance at different latitudes should account for curvature
        x1, y1 = coordinate_mapper.coord_to_metric(0.0, 0.0)
        x2, y2 = coordinate_mapper.coord_to_metric(1.0, 0.0)

        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # For small angles, distance ~ angle * R
        expected_order = coordinate_mapper.R * np.radians(1.0)
        assert 0.5 * expected_order < distance < 1.5 * expected_order

    def test_coordinate_symmetry(self, coordinate_mapper):
        """Test that transformations are symmetric around origin."""
        x_pos, y_pos = coordinate_mapper.coord_to_metric(0.1, 0.0)
        x_neg, y_neg = coordinate_mapper.coord_to_metric(-0.1, 0.0)

        assert np.isclose(x_pos, -x_neg, rtol=1e-5)
        assert np.isclose(y_pos, y_neg, rtol=1e-5)

    @pytest.mark.parametrize(
        "lon,lat",
        [
            (0.0, 0.0),
            (0.05, 0.05),
            (-0.05, -0.05),
            (0.1, -0.05),
            (-0.1, 0.05),
        ],
    )
    def test_round_trip_parametrized(self, coordinate_mapper, lon, lat):
        """Parametrized test for multiple coordinate pairs."""
        x_m, y_m = coordinate_mapper.coord_to_metric(lon, lat)
        lon_back, lat_back = coordinate_mapper.metric_to_coord(x_m, y_m)

        assert np.isclose(lon_back, lon, atol=1e-6)
        assert np.isclose(lat_back, lat, atol=1e-6)
