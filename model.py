"""Build the configured weekly auto-regressive model."""

import logging
import os

import optax
import pandas as pd
from chap_auto_regressive import AutoRegressiveModel
from chap_auto_regressive.model import FlaxPredictor

logger = logging.getLogger(__name__)

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
    "seed_offset",
    "early_stopping",
    "validation_periods",
    "patience",
    "eval_every",
    "min_probe_windows",
    "refit_scaling",
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
    "lr_schedule",
    "peak_learning_rate",
    "warmup_epochs",
)

#: Learning-rate schedules a run may select via the ``lr_schedule`` option.
LR_SCHEDULES = ("constant", "warmup_cosine")


class WeeklyAutoRegressiveModel(AutoRegressiveModel):
    """``AutoRegressiveModel`` with an optional warmup + cosine-decay learning rate.

    With ``lr_schedule="constant"`` (the default) training is unchanged: Adam at a
    fixed ``learning_rate``. With ``"warmup_cosine"`` the rate rises linearly from
    ``learning_rate`` to ``peak_learning_rate`` over ``warmup_epochs`` epochs, then
    follows a cosine decay back down to ``learning_rate`` at the end of ``n_iter``
    epochs. The decay length tracks ``n_iter``, so changing the epoch budget
    still gives a full decay.
    """

    # Re-declared with a type (same default as the library): during a scheduled
    # training run it temporarily holds the optax schedule.
    learning_rate: float | optax.Schedule = 1e-3
    lr_schedule: str = "constant"
    peak_learning_rate: float = 1e-3
    warmup_epochs: int = 5

    def learning_rate_for(self, steps_per_epoch: int) -> float | optax.Schedule:
        """Return the Adam learning rate (a constant or a per-step schedule).

        Args:
            steps_per_epoch: Optimizer steps per epoch (one per training window).
        """
        base_rate = self.learning_rate
        assert not callable(base_rate), "learning_rate is already a schedule"
        if self.lr_schedule == "constant":
            return base_rate
        if self.lr_schedule != "warmup_cosine":
            raise ValueError(f"unknown lr_schedule {self.lr_schedule!r}; expected one of {LR_SCHEDULES}")
        total_steps = max(1, self.n_iter * steps_per_epoch)
        warmup_steps = min(self.warmup_epochs * steps_per_epoch, total_steps - 1)
        return optax.warmup_cosine_decay_schedule(
            init_value=base_rate,
            peak_value=self.peak_learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=base_rate,
        )

    def train(self, data: pd.DataFrame) -> FlaxPredictor:
        """Train with the configured learning-rate schedule (see the class docstring)."""
        if self.lr_schedule == "constant":
            return super().train(data)
        # The library's Trainer hands ``learning_rate`` straight to ``optax.adam``,
        # which accepts a schedule in place of a float. The schedule counts
        # optimizer steps, and the Trainer takes one step per window.
        steps_per_epoch = len(self._get_dataset(data))
        base_rate = self.learning_rate
        self.learning_rate = self.learning_rate_for(steps_per_epoch)
        logger.info(
            "Learning-rate schedule %s: %s -> %s over %d warmup epochs, cosine decay to %s at epoch %d",
            self.lr_schedule,
            base_rate,
            self.peak_learning_rate,
            self.warmup_epochs,
            base_rate,
            self.n_iter,
        )
        try:
            return super().train(data)
        finally:
            self.learning_rate = base_rate


def build_model(options: dict | None = None) -> WeeklyAutoRegressiveModel:
    """Return the model, applying the weekly defaults and any run ``user_option_values``.

    The weekly base configuration (52-week context, 12-week horizon, lr 1e-5) is
    applied first; ``options`` (a run's ``user_option_values``) override individual
    knobs, and unknown keys are ignored with a warning. ``AR_N_ITER`` still
    overrides the epoch count so the test suite can run a fast pass.
    """
    options = {**_WEEKLY_DEFAULTS, **(options or {})}
    if "AR_N_ITER" in os.environ:
        options["n_iter"] = int(os.environ["AR_N_ITER"])
    model = WeeklyAutoRegressiveModel()
    # Early stopping is opt-in. It holds out a trailing validation block and needs
    # a long series (the library raises when the history is too short), so a run
    # must ask for it explicitly rather than inherit it from the library default.
    model.early_stopping = False
    for key, value in options.items():
        if key in MODEL_OPTIONS:
            setattr(model, key, value)
        else:
            logger.warning("Ignoring unknown model option: %s", key)
    return model
