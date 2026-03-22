"""Concrete implementations of BathymetryProvider."""

import functools
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .. import utils
from .base import BathymetryProvider


class FlatBathymetry(BathymetryProvider):
    """Uniform depth bathymetry."""

    def __init__(self, depth: float = -10.0):
        """
        Create flat bathymetry.

        Parameters
        ----------
        depth : float, default=-10.0
            Uniform depth in meters (negative below sea level)
        """
        self.depth = depth

    def get_bathymetry(
        self,
        lon: npt.NDArray[np.float64],
        lat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return array with uniform depth.

        Parameters
        ----------
        lon : np.ndarray
            Longitude meshgrid of shape (ny, nx)
        lat : np.ndarray
            Latitude meshgrid of shape (ny, nx)

        Returns
        -------
        np.ndarray
            Array of shape (ny, nx) with constant depth values
        """
        return self.depth * np.ones_like(lon)


class SlopingBathymetry(BathymetryProvider):
    """Bathymetry that slopes gradually in one direction."""

    def __init__(self, depth_min: float = -5.0, depth_max: float = -20.0):
        """
        Create sloping bathymetry.

        Parameters
        ----------
        depth_min : float, default=-5.0
            Shallowest depth (m), at y=0
        depth_max : float, default=-20.0
            Deepest depth (m), at y=max
        """
        self.depth_min = depth_min
        self.depth_max = depth_max

    def get_bathymetry(
        self,
        lon: npt.NDArray[np.float64],
        lat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return linearly sloping bathymetry in y-direction.

        Parameters
        ----------
        lon : np.ndarray
            Longitude meshgrid of shape (ny, nx)
        lat : np.ndarray
            Latitude meshgrid of shape (ny, nx)

        Returns
        -------
        np.ndarray
            Array of shape (ny, nx) with linearly varying depth
        """
        # Normalize latitude to [0, 1]
        lat_min, lat_max = np.min(lat), np.max(lat)
        lat_normalized = (lat - lat_min) / (lat_max - lat_min)

        bathymetry = self.depth_min + (self.depth_max - self.depth_min) * lat_normalized
        return bathymetry


class BathymetryFromNC(BathymetryProvider):
    """Bathymetry loaded from a NetCDF file using interpolation."""

    def __init__(self, nc_path: str | Path):
        """
        Create bathymetry provider from a NetCDF file.

        Parameters
        ----------
        nc_path : str | Path
            Path to the NetCDF file containing bathymetry data
        """
        self.nc_path = Path(nc_path)

        self.bathymetry_interpolator = functools.partial(
            utils.bathymetry.interpolate_gebco_on_grid,
            nc_path=self.nc_path,
        )

    def get_bathymetry(
        self,
        lon: npt.NDArray[np.float64],
        lat: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return bathymetry interpolated from NetCDF file.

        Parameters
        ----------
        lon : np.ndarray
            Longitude meshgrid of shape (ny, nx)
        lat : np.ndarray
            Latitude meshgrid of shape (ny, nx)

        Returns
        -------
        np.ndarray
            Array of shape (ny, nx) with bathymetry values from file
        """
        bathymetry_values = self.bathymetry_interpolator(X=lon, Y=lat)
        bathymetry_values[np.isnan(bathymetry_values)] = 0.0
        return bathymetry_values
