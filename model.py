"""Build the configured weekly auto-regressive model."""

import logging
import os

import pandas as pd
from chap_auto_regressive import AutoRegressiveModel
from chap_auto_regressive.transforms import REQUIRED_COVARIATES

logger = logging.getLogger(__name__)

# Index/target/identifier columns that are never covariates.
_NON_COVARIATE_COLUMNS = frozenset({"time_period", "location", "disease_cases", "parent"})

# Weekly base configuration, applied before any run options. The library ships
# monthly-tuned defaults (3-period horizon, 36-period context); the weekly model
# reads a year of weekly history (52) and forecasts a quarter ahead (12).
_WEEKLY_DEFAULTS = {
    "context_length": 52,
    "prediction_length": 12,
    "learning_rate": 1e-5,
    "n_iter": 1000,
}

# Model options a run may set via user_option_values (all optional; defaults are
# the weekly configuration above layered on the AutoRegressiveModel defaults).
MODEL_OPTIONS = (
    "n_iter",
    "context_length",
    "n_ensemble",
    "learning_rate",
    "prediction_length",
    "cell",
    "rnn_features",
    "head_features",
    "rnn_layers",
    "dropout_rate",
    "recursive_decode",
    "input_dropout_rate",
)


def additional_covariates(data: pd.DataFrame) -> list[str]:
    """Return the additional covariate columns present in a training frame.

    Beyond the index, target and required covariates, CHAP includes the
    ``additional_continuous_covariates`` named in the run config -- every such
    column is fed to the network as an extra feature, in column order. Only
    numeric columns qualify, which skips the string ``parent``/``location``
    identifiers and the unnamed index CHAP writes into the CSV.
    """
    skip = _NON_COVARIATE_COLUMNS | set(REQUIRED_COVARIATES)
    return [
        c
        for c in data.columns
        if c not in skip and not str(c).startswith("Unnamed") and pd.api.types.is_numeric_dtype(data[c])
    ]


def build_model(options: dict | None = None) -> AutoRegressiveModel:
    """Return the model, applying the weekly defaults and any run ``user_option_values``.

    The weekly base configuration (52-week context, 12-week horizon, lr 1e-5) is
    applied first; ``options`` (a run's ``user_option_values``) override individual
    knobs, and unknown keys are ignored with a warning. ``AR_N_ITER`` still
    overrides the epoch count so the test suite can run a fast pass.
    """
    options = {**_WEEKLY_DEFAULTS, **(options or {})}
    if "AR_N_ITER" in os.environ:
        options["n_iter"] = int(os.environ["AR_N_ITER"])
    model = AutoRegressiveModel()
    for key, value in options.items():
        if key in MODEL_OPTIONS:
            setattr(model, key, value)
        else:
            logger.warning("Ignoring unknown model option: %s", key)
    return model
