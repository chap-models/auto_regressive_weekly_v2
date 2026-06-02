"""Build the configured weekly auto-regressive model."""

import os

from chap_auto_regressive import AutoRegressiveModel


def build_model() -> AutoRegressiveModel:
    """Return the configured weekly model.

    ``AR_N_ITER`` overrides the training-iteration count (default 1000) so the
    test suite can run a fast pass.
    """
    model = AutoRegressiveModel()
    model.n_iter = int(os.environ.get("AR_N_ITER", "1000"))
    model.context_length = 52
    model.prediction_length = 12
    model.learning_rate = 1e-5
    return model
