"""Setup script for swe_simulator package."""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        packages=find_packages(),
        package_dir={"swe_simulator": "swe_simulator"},
    )
