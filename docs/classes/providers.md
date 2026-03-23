# Providers

Providers are small classes that supply simulation inputs to `SWESolver`.

Instead of hardcoding equations/data loading inside the solver, each concern is
delegated to a provider:

- initial condition provider: returns `(h, hu, hv)` at $t=0$
- bathymetry provider: returns seabed elevation/depth
- wind provider: returns wind field `(u_wind, v_wind)` (possibly time-dependent)

This keeps the solver focused on numerics and orchestration.

## Why this pattern is useful

- Swap scenarios without changing solver internals
- Reuse the same solver with synthetic or real datasets
- Keep classes testable (providers are easy to unit test in isolation)
- Add new physics/data sources by adding a class, not rewriting solver logic

## Provider interfaces

The project uses three abstract interfaces:

- `InitialConditionProvider`
- `BathymetryProvider`
- `WindProvider`

Any custom provider must implement the corresponding method signature used by
the solver.

## Built-in providers

### Initial condition providers

- `GaussianHumpInitialCondition`: Gaussian perturbation for wave tests
- `FlatInitialCondition`: uniform still-water initial state

Example:

```python
from swe_simulator.providers import GaussianHumpInitialCondition

ic_provider = GaussianHumpInitialCondition(
	height=2.0,
	width=0.01,
	center=(0.0, 0.0),
	water_velocity=(0.0, 0.0),
)
```

### Bathymetry providers

- `FlatBathymetry`: constant depth everywhere
- `SlopingBathymetry`: linear depth variation (useful for coastal tests)
- `BathymetryFromNC`: interpolates bathymetry from NetCDF (e.g., GEBCO)

Example:

```python
from swe_simulator.providers import BathymetryFromNC

bathy_provider = BathymetryFromNC(
	nc_path="data/gebco_2025_n25.9288_s25.6527_w-80.2016_e-80.0642.nc"
)
```

### Wind providers

- `ConstantWind`: uniform, time-independent wind field

Example:

```python
from swe_simulator.providers import ConstantWind

wind_provider = ConstantWind(u_wind=5.0, v_wind=2.0)
```

## Using providers with `SWESolver`

```python
from swe_simulator.config import SimulationConfig
from swe_simulator.providers import (
	BathymetryFromNC,
	ConstantWind,
	GaussianHumpInitialCondition,
)
from swe_simulator.solver import SWESolver

config = SimulationConfig(
	lon_range=(-80.1865, -80.0791),
	lat_range=(25.6678, 25.9137),
	nx=40,
	ny=40,
	t_final=1000.0,
	dt=1.0,
)

solver = SWESolver(
	config=config,
	ic_provider=GaussianHumpInitialCondition(height=2.0, width=0.01),
	bathymetry_provider=BathymetryFromNC(
		"data/gebco_2025_n25.9288_s25.6527_w-80.2016_e-80.0642.nc"
	),
	wind_provider=ConstantWind(u_wind=5.0, v_wind=2.0),
)

solver.setup_solver()
result = solver.solve()
```

## Writing a custom provider

If you need custom logic, inherit from the base interface and implement the
required method.

Example custom wind provider:

```python
import numpy as np
from swe_simulator.providers import WindProvider


class SinusoidalWind(WindProvider):
	def __init__(self, amplitude: float = 8.0, period: float = 3600.0):
		self.amplitude = amplitude
		self.period = period

	def get_wind(self, lon: np.ndarray, lat: np.ndarray, time: float):
		u = self.amplitude * np.sin(2 * np.pi * time / self.period)
		v = 0.0
		return u * np.ones_like(lon), v * np.ones_like(lat)
```

Then pass it to `SWESolver(wind_provider=SinusoidalWind(...))`.
