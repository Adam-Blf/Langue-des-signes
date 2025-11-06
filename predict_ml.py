from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import joblib
import numpy as np

LOGGER = logging.getLogger(__name__)
MODEL_PATH = Path(__file__).resolve().parent / "machine_learning" / "model.pkl"
_MODEL_CACHE = None
_MODEL_LOAD_FAILED = False


def load_model(force_reload: bool = False):
    """Return the cached scikit-learn model, loading it from disk on demand."""
    global _MODEL_CACHE, _MODEL_LOAD_FAILED

    if _MODEL_CACHE is not None and not force_reload:
        return _MODEL_CACHE

    if not MODEL_PATH.exists():
        _MODEL_LOAD_FAILED = True
        LOGGER.warning("ML model not found at %s", MODEL_PATH)
        return None

    try:
        _MODEL_CACHE = joblib.load(MODEL_PATH)
        _MODEL_LOAD_FAILED = False
    except Exception as exc:  # pragma: no cover - defensive logging
        _MODEL_CACHE = None
        _MODEL_LOAD_FAILED = True
        LOGGER.exception("Unable to load ML model at %s", MODEL_PATH)
        raise RuntimeError("Failed to load ML model") from exc

    return _MODEL_CACHE


def _prepare_landmarks(landmarks: Iterable[float]) -> Optional[np.ndarray]:
    """Return a numpy array with 63 values or ``None`` if the input is invalid."""
    if isinstance(landmarks, np.ndarray):
        arr = landmarks
    elif isinstance(landmarks, Sequence):
        arr = np.asarray(landmarks, dtype=np.float32)
    else:
        arr = np.fromiter((float(value) for value in landmarks), dtype=np.float32)

    if arr.size != 63:
        LOGGER.debug("Unexpected landmark vector size: %s", arr.size)
        return None

    return arr.reshape(1, -1)


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


def predict_ml_with_confidence(
    landmarks: Iterable[float], threshold: float = 0.83
) -> Optional[Prediction]:
    """Predict a letter and confidence when above *threshold*."""
    if _MODEL_LOAD_FAILED:
        return None

    model = load_model()
    if model is None:
        return None

    coords = _prepare_landmarks(landmarks)
    if coords is None:
        return None

    probabilities = model.predict_proba(coords)[0]
    max_probability = float(np.max(probabilities))
    if max_probability < threshold:
        return None

    label = str(model.classes_[int(np.argmax(probabilities))])
    return Prediction(label=label, confidence=max_probability)


def predict_ml(landmarks: Iterable[float], threshold: float = 0.83) -> Optional[str]:
    """Compatibility helper that keeps the original signature."""
    prediction = predict_ml_with_confidence(landmarks, threshold)
    return prediction.label if prediction else None
