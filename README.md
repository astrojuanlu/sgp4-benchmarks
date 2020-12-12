# SGP4 benchmarks

## Installation

Install the dependencies:

```
(env) $ python -m pip install -r requirements.txt
```

or, alternatively:

```
(env) $ python -m pip install pip-tools && pip-sync
```

## Running the benchmarks

To run the benchmarks:

```
(env) $ pytest
```

and, to produce the histogram plots,

```
(env) $ pytest --benchmark-histogram
```

### Slow benchmarks

The "multiple satellites, multiple dates" large case is extremely slow compared to the other ones,
and this is especially troublesome for the pure Python implementation.
If you want to disable the slowest benchmarks, use the `-m "not slow"` option in pytest.
Example:

```
(env) $ pytest -k pure -m "not slow"
```
