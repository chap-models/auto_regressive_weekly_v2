"""CHAP train entry point: ``python train.py <train_data.csv> <model_out> [--config cfg.yaml]``."""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from model import additional_covariates, build_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_user_options(config_path: str | None) -> dict:
    """Read ``user_option_values`` from a CHAP model-configuration YAML, if given."""
    if not config_path or not Path(config_path).exists():
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    return config.get("user_option_values", {}) or {}


def main() -> None:
    """Train the model on a CSV and save the predictor."""
    parser = argparse.ArgumentParser(description="Train the auto-regressive model")
    parser.add_argument("train_data", help="path to the training data CSV")
    parser.add_argument("model", help="path to write the trained model")
    parser.add_argument("--config", default=None, help="path to a CHAP model-configuration YAML")
    args = parser.parse_args()

    data = pd.read_csv(args.train_data)
    options = _read_user_options(args.config)
    if options:
        logger.info("Applying model options: %s", options)
    model = build_model(options)
    # Any covariate column beyond the required three is fed to the network as an
    # additional feature. The chosen covariates are persisted in the saved
    # predictor, so predict.py needs no matching configuration.
    model.additional_covariates = additional_covariates(data)
    if model.additional_covariates:
        logger.info("Using additional covariates: %s", model.additional_covariates)
    predictor = model.train(data)
    predictor.save(args.model)


if __name__ == "__main__":
    main()
