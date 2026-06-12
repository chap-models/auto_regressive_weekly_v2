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
import yaml

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "input" / "trainData.csv"
PREDICTION_LENGTH = 12


def _write_config(tmp_path: Path, **options) -> str:
    """Write a CHAP model-configuration YAML with the given user_option_values."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"user_option_values": options}))
    return str(path)


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


def test_additional_covariates_helper_skips_index_target_required_and_chap_metadata():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))  # model.py lives at the repo root
    from model import additional_covariates

    # CHAP writes an unnamed index and a string `parent` column alongside the
    # declared covariates; only the numeric extra covariates should be selected.
    frame = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "time_period": ["2020-W01", "2020-W02"],
            "location": ["A", "A"],
            "parent": ["P", "P"],
            "disease_cases": [1.0, 2.0],
            "rainfall": [1.0, 2.0],
            "mean_temperature": [20.0, 21.0],
            "population": [1000.0, 1000.0],
            "relative_humidity": [50.0, 55.0],
            "irs_decay": [0.1, 0.2],
        }
    )
    assert additional_covariates(frame) == ["relative_humidity", "irs_decay"]


def test_train_then_predict_with_additional_covariate(tmp_path: Path) -> None:
    # An extra covariate column in the data is fed to the network at train time
    # and required at predict time (it is persisted in the saved model).
    env = {**os.environ, "AR_N_ITER": "30"}
    df = pd.read_csv(INPUT)
    rng = np.random.RandomState(0)
    df["relative_humidity"] = rng.rand(len(df)) * 100  # a derived extra covariate

    data_path = tmp_path / "train_extra.csv"
    df.to_csv(data_path, index=False)
    model_path = tmp_path / "model.bin"
    cfg = _write_config(tmp_path, n_ensemble=1)
    subprocess.run(
        [sys.executable, "train.py", str(data_path), str(model_path), "--config", cfg], cwd=REPO, env=env, check=True
    )

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
