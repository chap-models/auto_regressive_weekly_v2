"""CHAP train entry point: ``python train.py <train_data.csv> <model_out> [--config cfg.yaml]``."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from chap_auto_regressive import AutoRegressiveModel

from model import build_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_config(config_path: str | None) -> dict:
    """Read a CHAP model-configuration YAML, if given."""
    if not config_path or not Path(config_path).exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _clamp_context_length(model: AutoRegressiveModel, data: pd.DataFrame) -> None:
    """Shrink ``context_length`` so the training data yields at least one window.

    Training needs ``context_length + prediction_length`` periods per window and
    prediction needs ``context_length`` periods of history, so the largest safe
    context is ``shortest history - prediction_length``. The clamped value is
    persisted in the trained predictor, so predict stays consistent.
    """
    period_counts: list[int] = data.groupby("location")["time_period"].nunique().tolist()
    periods = min(period_counts)
    max_context = periods - model.prediction_length
    if max_context < 1:
        raise ValueError(
            f"training data is too short: the shortest location history has {periods} periods, "
            f"but at least prediction_length+1={model.prediction_length + 1} are needed"
        )
    if model.context_length > max_context:
        logger.warning(
            "context_length=%d exceeds the available history (%d periods); clamping to %d",
            model.context_length,
            periods,
            max_context,
        )
        model.context_length = max_context


def main() -> None:
    """Train the model on a CSV and save the predictor."""
    parser = argparse.ArgumentParser(description="Train the auto-regressive model")
    parser.add_argument("train_data", help="path to the training data CSV")
    parser.add_argument("model", help="path to write the trained model")
    parser.add_argument("--config", default=None, help="path to a CHAP model-configuration YAML")
    args = parser.parse_args()

    data = pd.read_csv(args.train_data)
    config = _read_config(args.config)
    options = config.get("user_option_values") or {}
    if options:
        logger.info("Applying model options: %s", options)
    model = build_model(options)
    _clamp_context_length(model, data)
    # Only the covariates the configuration declares are fed to the network on
    # top of the required three. CHAP writes every dataset column into the
    # training CSV, so the columns present do not decide what is used. The chosen
    # covariates are persisted in the saved predictor, so predict.py needs no
    # matching configuration.
    model.additional_covariates = list(config.get("additional_continuous_covariates") or [])
    if model.additional_covariates:
        logger.info("Using additional covariates: %s", model.additional_covariates)
    predictor = model.train(data)
    predictor.save(args.model)


if __name__ == "__main__":
    main()
