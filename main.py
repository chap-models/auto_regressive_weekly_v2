"""CHAP entry point for the weekly deep auto-regressive model."""

import logging
import os

from chap_ar import AutoRegressiveModel
from chap_core.adaptors.command_line_interface import generate_app

logging.basicConfig(level=logging.INFO)

model = AutoRegressiveModel()
# Training iterations default to 1000; AR_N_ITER lets tests run a fast pass.
model.n_iter = int(os.environ.get("AR_N_ITER", "1000"))
model.context_length = 52
model.prediction_length = 12
model.learning_rate = 1e-5

app = generate_app(model)
app()
