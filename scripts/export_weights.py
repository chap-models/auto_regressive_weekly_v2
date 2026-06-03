"""Export a trained model to a portable format for the Rust inference port.

Reads a predictor pickle written by ``train.py`` and writes a directory the
``chap-ar-predict`` Rust binary can load:

- ``weights.safetensors`` -- the network parameters, keyed by their flattened Flax
  pytree path (e.g. ``params/preprocess/Dense_0/kernel``).
- ``meta.json`` -- the context/prediction lengths, architecture name, embedding
  dimension, feature order, the fitted scaler ``mu``/``std`` and the sorted
  training locations.

The context/prediction lengths and architecture are taken from ``build_model()``,
so they always match how this wrapper trains -- no need to pass them by hand.

Usage:
    uv run python scripts/export_weights.py <model> <out_dir>
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from flax.traverse_util import flatten_dict
from safetensors.numpy import save_file

# Allow `from model import build_model` when run as `python scripts/export_weights.py`
# from the project root (model.py lives one level up, not in scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import build_model

# The fixed per-period feature order produced by transforms.get_series.
FEATURE_ORDER = ["rainfall", "mean_temperature", "population", "year_position"]


def _load_pickle(path: str) -> tuple:
    """Load the ``(params, scaler, locations)`` tuple written by ``save``."""
    if not Path(path).is_file():
        raise SystemExit(f"no such model file: {path!r} (train one first with train.py)")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if len(payload) == 2:  # legacy pickle without tracked locations
        params, scaler = payload
        return params, scaler, None
    params, scaler, locations = payload
    return params, scaler, locations


def main() -> None:
    """Export the trained model into ``out_dir``."""
    parser = argparse.ArgumentParser(description="Export a trained model for the Rust port")
    parser.add_argument("model", help="path to a model written by train.py")
    parser.add_argument("out_dir", help="directory to write weights.safetensors and meta.json into")
    args = parser.parse_args()

    config = build_model()
    params, scaler, locations = _load_pickle(args.model)
    tensors = {key: np.asarray(value, dtype=np.float32) for key, value in flatten_dict(params, sep="/").items()}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "weights.safetensors"))

    embedding = tensors["params/preprocess/Embed_0/embedding"]
    meta = {
        "rnn_model_name": config.rnn_model_name,
        "context_length": config.context_length,
        "prediction_length": config.prediction_length,
        "n_locations": int(embedding.shape[0]),
        "embedding_dim": int(embedding.shape[1]),
        "feature_order": FEATURE_ORDER,
        "scaler": {
            "mu": [float(v) for v in np.asarray(scaler.mu).ravel()],
            "std": [float(v) for v in np.asarray(scaler.std).ravel()],
        },
        "locations": list(locations) if locations is not None else None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"wrote {out/'weights.safetensors'} ({len(tensors)} tensors) and {out/'meta.json'} "
        f"(context={config.context_length}, prediction={config.prediction_length})"
    )


if __name__ == "__main__":
    main()
