# WindForcing

`WindForcing` applies wind stress source terms to momentum equations.

## Inputs

- `mesgrid_domain`: `(X_coord, Y_coord)` meshgrids
- `wind_provider`: object implementing wind retrieval
- `c_d`, `rho_air`, `rho_water`: drag and densities

## Responsibilities

- Read wind via provider (`get_wind`)
- Compute water velocities from `(h, hu, hv)`
- Compute wind stress `(tau_x, tau_y)`
- Apply source update in `__call__` for PyClaw

## Usage

Usually created internally by `SWESolver.initialize_data_from_providers()`.
