"""Self-contained pipeline test: run train.py + predict.py on the bundled data.

This needs no chap-core -- it drives the model's own CLI scripts directly and
checks the prediction CSV. The deeper chap eval integration lives in
``test_eval.py`` (skipped unless the ``chap`` CLI is available).
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "input" / "trainData.csv"
PREDICTION_LENGTH = 12


def _write_config(tmp_path: Path, covariates: list[str] | None = None, **options) -> str:
    """Write a CHAP model-configuration YAML with the given covariates and user_option_values."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"additional_continuous_covariates": covariates or [], "user_option_values": options})
    )
    return str(path)


def _train_and_predict(tmp_path: Path, train_df: pd.DataFrame, predict_df: pd.DataFrame, cfg: str) -> pd.DataFrame:
    """Train on ``train_df``, forecast the last periods of ``predict_df``, and return the predictions."""
    env = {**os.environ, "AR_N_ITER": "30"}
    data_path = tmp_path / "train.csv"
    train_df.to_csv(data_path, index=False)
    model_path = tmp_path / "model.bin"
    subprocess.run(
        [sys.executable, "train.py", str(data_path), str(model_path), "--config", cfg], cwd=REPO, env=env, check=True
    )

    future = pd.concat(
        [
            sub.sort_values("time_period").tail(PREDICTION_LENGTH).drop(columns=["disease_cases"])
            for _, sub in predict_df.groupby("location", sort=False)
        ],
        ignore_index=True,
    )
    historic_path = tmp_path / "historic.csv"
    future_path = tmp_path / "future.csv"
    out_path = tmp_path / "predictions.csv"
    predict_df.to_csv(historic_path, index=False)
    future.to_csv(future_path, index=False)

    subprocess.run(
        [sys.executable, "predict.py", str(model_path), str(historic_path), str(future_path), str(out_path)],
        cwd=REPO,
        env=env,
        check=True,
    )
    return pd.read_csv(out_path)


def test_train_then_predict(tmp_path: Path) -> None:
    env = {**os.environ, "AR_N_ITER": "30"}  # fast pass; production default is 1000
    model_path = tmp_path / "model.bin"
    # A single ensemble member keeps the test fast; the default is a 5-member ensemble.
    cfg = _write_config(tmp_path, n_ensemble=1)

    subprocess.run(
        [sys.executable, "train.py", str(INPUT), str(model_path), "--config", cfg], cwd=REPO, env=env, check=True
    )

    # Build historic (full series) and future (last `prediction_length` periods,
    # covariates only) inputs from the bundled data.
    df = pd.read_csv(INPUT)
    future = pd.concat(
        [
            sub.sort_values("time_period").tail(PREDICTION_LENGTH).drop(columns=["disease_cases"])
            for _, sub in df.groupby("location", sort=False)
        ],
        ignore_index=True,
    )
    historic_path = tmp_path / "historic.csv"
    future_path = tmp_path / "future.csv"
    out_path = tmp_path / "predictions.csv"
    df.to_csv(historic_path, index=False)
    future.to_csv(future_path, index=False)

    subprocess.run(
        [sys.executable, "predict.py", str(model_path), str(historic_path), str(future_path), str(out_path)],
        cwd=REPO,
        env=env,
        check=True,
    )

    out = pd.read_csv(out_path)
    sample_cols = [c for c in out.columns if c.startswith("sample_")]
    assert {"time_period", "location"}.issubset(out.columns)
    assert sample_cols, "no sample_* columns in the output"
    assert np.isfinite(out[sample_cols].to_numpy()).all()
    assert len(out) == df["location"].nunique() * PREDICTION_LENGTH


def test_train_then_predict_with_additional_covariate(tmp_path: Path) -> None:
    # A covariate declared in the configuration is fed to the network at train
    # time and required at predict time (it is persisted in the saved model).
    df = pd.read_csv(INPUT)
    rng = np.random.RandomState(0)
    df["relative_humidity"] = rng.rand(len(df)) * 100  # a derived extra covariate

    cfg = _write_config(tmp_path, covariates=["relative_humidity"], n_ensemble=1)
    out = _train_and_predict(tmp_path, df, df, cfg)

    sample_cols = [c for c in out.columns if c.startswith("sample_")]
    assert sample_cols and np.isfinite(out[sample_cols].to_numpy()).all()
    assert len(out) == df["location"].nunique() * PREDICTION_LENGTH


def test_undeclared_covariate_column_is_not_used(tmp_path: Path) -> None:
    # CHAP writes every dataset column into the training CSV. A numeric column the
    # configuration does not declare must not reach the network, so predicting
    # without it has to work.
    df = pd.read_csv(INPUT)
    train_df = df.assign(relative_humidity=np.random.RandomState(0).rand(len(df)) * 100)

    cfg = _write_config(tmp_path, n_ensemble=1)
    out = _train_and_predict(tmp_path, train_df, df, cfg)

    sample_cols = [c for c in out.columns if c.startswith("sample_")]
    assert sample_cols and np.isfinite(out[sample_cols].to_numpy()).all()
    assert len(out) == df["location"].nunique() * PREDICTION_LENGTH


def test_build_model_applies_user_options_and_ignores_unknown():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))  # model.py lives at the repo root
    from model import build_model

    model = build_model({"context_length": 18, "n_ensemble": 3, "cell": "simple", "bogus_option": 1})
    assert model.context_length == 18
    assert model.n_ensemble == 3
    assert model.cell == "simple"  # architecture knob applied
    assert not hasattr(model, "bogus_option")  # unknown ignored, not set


def test_build_model_defaults_to_weekly_configuration():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))  # model.py lives at the repo root
    from model import build_model

    model = build_model()
    assert model.context_length == 52  # a year of weekly history
    assert model.prediction_length == 12  # a quarter-ahead horizon


def test_warmup_cosine_schedule_shape():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))  # model.py lives at the repo root
    from model import build_model

    model = build_model({"lr_schedule": "warmup_cosine", "n_iter": 100, "learning_rate": 1e-5})
    steps_per_epoch = 10
    schedule = model.learning_rate_for(steps_per_epoch)
    assert callable(schedule)
    assert np.isclose(schedule(0), 1e-5)  # starts at the base rate
    assert np.isclose(schedule(5 * steps_per_epoch), 1e-3)  # peak after the 5 default warmup epochs
    assert np.isclose(schedule(100 * steps_per_epoch), 1e-5)  # decayed back by the end of n_iter
    assert np.isclose(schedule(1000 * steps_per_epoch), 1e-5)  # and stays there


def test_constant_schedule_is_the_default_and_unknown_schedule_fails():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))  # model.py lives at the repo root
    from model import build_model

    assert build_model().learning_rate_for(10) == 1e-5
    with pytest.raises(ValueError, match="lr_schedule"):
        build_model({"lr_schedule": "cosine"}).learning_rate_for(10)


def test_train_then_predict_with_warmup_cosine_schedule(tmp_path: Path) -> None:
    env = {**os.environ, "AR_N_ITER": "10"}
    model_path = tmp_path / "model.bin"
    cfg = _write_config(tmp_path, n_ensemble=1, lr_schedule="warmup_cosine", warmup_epochs=2)
    result = subprocess.run(
        [sys.executable, "train.py", str(INPUT), str(model_path), "--config", cfg],
        cwd=REPO,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Learning-rate schedule warmup_cosine" in result.stderr
    assert model_path.exists()


def test_early_stopping_is_opt_in():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))  # model.py lives at the repo root
    from model import build_model

    assert build_model().early_stopping is False
    assert build_model({"early_stopping": True, "validation_periods": 8}).validation_periods == 8


def test_context_length_is_clamped_to_short_history(tmp_path: Path) -> None:
    # 40 weeks is too short for the 52-week context plus the 12-week horizon, so
    # train.py shrinks the context instead of failing.
    df = pd.read_csv(INPUT)
    periods = sorted(df["time_period"].unique())[-40:]
    short = df.loc[df["time_period"].isin(periods)].reset_index(drop=True)
    out = _train_and_predict(tmp_path, short, short, _write_config(tmp_path, n_ensemble=1))

    sample_cols = [c for c in out.columns if c.startswith("sample_")]
    assert sample_cols and np.isfinite(out[sample_cols].to_numpy()).all()
