# auto_regressive_weekly

Weekly Deep Auto Regressive model for CHAP. An experimental deep learning model
based on an RNN architecture that forecasts disease cases from auto-regressive
time series data and climate covariates.

It wraps `AutoRegressiveModel` from
[`chap_ar`](https://github.com/mortenoh/chap_ar) and exposes the standard CHAP
`train` / `predict` command-line interface via
`chap_core.adaptors.command_line_interface.generate_app`.

## Configuration

| setting | value |
| --- | --- |
| period type | week |
| context length | 52 |
| prediction length | 12 |
| training iterations | 1000 |
| learning rate | 1e-5 |
| required covariates | rainfall, mean_temperature, population |

## Environment

This model uses [uv](https://docs.astral.sh/uv/) and Python 3.13. The pinned
environment lives in `pyproject.toml` / `uv.lock`. CHAP runs the model through
its uv runner (`uv run python main.py ...`); the committed lock file makes
environment creation deterministic and fast.

Key pins:

- Python 3.13
- `chap-core` @ git `master` (dhis2-chap/chap-core)
- `chap_ar` @ git (mortenoh/chap_ar) — the deep AR flax model, providing `AutoRegressiveModel`
- `flax 0.12`, `jax 0.10` (resolved transitively via `chap_ar`)

The number of training iterations defaults to **1000**. Set the `AR_N_ITER`
environment variable to override it — CHAP passes it through to the model process,
so for example the test suite runs with `AR_N_ITER=30` to make a full `chap eval`
finish in a couple of minutes. Lower it for quick checks, leave it at the default
for production forecasts.

## Development

```bash
make install   # uv sync
make check     # ruff (format + lint) + mypy + pyright, no changes
make lint      # ruff format + autofix, then type-check
make test      # run the model through the chap CLI on bundled input data
```

`make test` requires the `chap` CLI (provided by chap-core, a dependency of this
model) and fails clearly if it is missing. It runs a small backtest via
`chap eval` against `input/trainData.csv` (a trimmed weekly dataset bundled in
`input/`) and asserts the forecasts are finite.

## Local use

```bash
uv sync

# train
uv run python main.py train <training_data.csv> <model_output_path>

# predict
uv run python main.py predict <model_path> <historic_data.csv> <future_data.csv> <out_file.csv>
```

## Evaluating through CHAP

`--model-name` can point straight at the GitHub repo; CHAP clones and runs it.
From a chap-core checkout (using a weekly dataset):

```bash
uv run chap eval \
    --model-name https://github.com/mortenoh/auto_regressive_weekly \
    --dataset-csv example_data/nicaragua_weekly_data.csv \
    --output-file /tmp/chap/ar_weekly_eval.nc \
    --backtest-params.n-splits 2 \
    --backtest-params.n-periods 1
```
