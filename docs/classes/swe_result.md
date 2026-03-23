# SWEResult

`SWEResult` is the simulation output dataclass returned by `SWESolver.solve()`.

## Initialization Arguments

The class is typically instantiated internally, but it accepts:

- `meshgrid_coord: tuple[np.ndarray | float, np.ndarray | float]`
- `meshgrid_metric: tuple[np.ndarray, np.ndarray]`
- `solution: np.ndarray`
- `bathymetry: np.ndarray`
- `initial_condition: np.ndarray`
- `wind_forcing: tuple[float | np.ndarray, float | np.ndarray]`
- `config: SimulationConfig`

## Attributes

### Geometry

- `meshgrid_coord`
  - Geographic grid (longitude, latitude).
- `meshgrid_metric`
  - Metric-space grid (x, y).

### Fields and State

- `solution`
  - Time-stacked SWE state array (one frame per output).
- `bathymetry`
- `initial_condition`
- `wind_forcing`

### Configuration

- `config`
  - `SimulationConfig` used for the run.

## Methods

### `to_dict() -> dict[str, Any]`

Returns dataclass content as a dictionary.

### `save(filepath: Path | str) -> None`

Serializes and writes the result object to disk using pickle.

### `load(filepath: Path | str) -> SWEResult`

Class method that reads a pickled result and returns a `SWEResult`.

## Example

```python
result = solver.solve()
result.save("_output/result.pkl")
loaded = SWEResult.load("_output/result.pkl")
```
