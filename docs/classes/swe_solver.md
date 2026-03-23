# SWESolver

`SWESolver` is the main entry point for building and running SWE simulations.

## Typical workflow

1. Create `SimulationConfig`
2. Instantiate `SWESolver`
3. Set data (providers or arrays)
4. Call `setup_solver()`
5. Call `solve()`

## Main methods

- `set_domain(lon_range, lat_range, nx, ny)`
- `set_time_parameters(t_final, dt)`
- `set_bathymetry(array)`
- `set_initial_condition(array)`
- `set_constant_wind_forcing(u_wind, v_wind)`
- `setup_solver()`
- `solve()`

## Notes

- `setup_solver()` configures the PyClaw solver, domain/state, and controller.
- `solve()` returns an `SWEResult` object and can write `result.pkl` to output.
