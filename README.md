# auto_regressive_weekly

Weekly Deep Auto Regressive model for CHAP. An experimental deep learning model
based on an RNN architecture that forecasts disease cases from auto-regressive
time series data and climate covariates.

It wraps `AutoRegressiveModel` from
[`chap_auto_regressive`](https://github.com/mortenoh/chap_auto_regressive) and exposes the standard CHAP
`train` / `predict` commands via small `train.py` / `predict.py` scripts that read
and write CSV with pandas. The model has no chap-core dependency at runtime.

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
its uv runner (`uv run python train.py …` / `predict.py …`); the committed lock file makes
environment creation deterministic and fast.

Key pins:

- Python 3.13
- `chap_auto_regressive` @ git (mortenoh/chap_auto_regressive) — the deep AR flax model, providing `AutoRegressiveModel`
- `flax 0.12`, `jax 0.10` (resolved transitively via `chap_auto_regressive`)

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
make test      # self-contained train/predict pipeline test
make eval      # chap eval backtest (chap CLI via uvx; no chap-core dependency)
```

This model has **no chap-core dependency**. Testing is split in two:

- `make test` is **self-contained** — it drives `train.py` / `predict.py` on the
  bundled `input/` data (a trimmed weekly dataset) and verifies the prediction CSV
  (columns, finiteness, row count). No chap-core involved.
- `make eval` runs a real `chap eval` backtest, then reads back the output NetCDF
  and asserts the forecasts are finite. It gets the `chap` CLI on demand with
  `uvx --from chap-core` — chap-core is never added to the project. CI runs both.

## Local use

```bash
uv sync

# train
uv run python train.py <training_data.csv> <model_output_path>

# predict
uv run python predict.py <model_path> <historic_data.csv> <future_data.csv> <out_file.csv>
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
