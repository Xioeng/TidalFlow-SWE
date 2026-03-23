# SWEResult

`SWEResult` stores simulation outputs and metadata.

## Stored fields

- Coordinate grids (geographic and metric)
- Time-stacked `solution` frames
- `bathymetry` and `initial_condition`
- `wind_forcing`
- `config`

## Methods

- `to_dict()`: dataclass serialization to dict
- `save(filepath)`: pickle serialization
- `load(filepath)`: restore serialized results

## Example

```python
result = solver.solve()
result.save("output/result.pkl")
loaded = result.load("output/result.pkl")
```
