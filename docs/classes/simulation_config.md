# SimulationConfig

`SimulationConfig` defines domain, time, physics, boundary conditions, and output behavior.

## Key fields

- `lon_range`, `lat_range`: geographic bounds
- `nx`, `ny`: grid resolution
- `t_final`, `dt`: integration horizon and time step
- `gravity`: physical constant
- `bc_lower`, `bc_upper`: boundary conditions in x/y
- `output_dir`, `multiple_output_times`: output policy

## Basic usage

```python
from swe_simulator.config import SimulationConfig

config = SimulationConfig(
    lon_range=(-80.2, -80.0),
    lat_range=(25.6, 25.9),
    nx=40,
    ny=40,
    t_final=1000.0,
    dt=1.0,
)
```

## Persistence

```python
config.save("config.json")
loaded = SimulationConfig.load("config.json")
```
