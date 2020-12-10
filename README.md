# SGP4 benchmarks

Install the dependencies:

```
(env) $ python -m pip install -r requirements.txt
```

or, alternatively:

```
(env) $ python -m pip install pip-tools && pip-sync
```

To run the benchmarks:

```
(env) $ pytest
```

and, to produce the histogram plots,

```
(env) $ pytest --benchmark-histogram
```
