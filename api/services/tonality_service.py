"""Tonality prediction service.

Loads pre-trained models to predict the tonality of input text using
sentence embeddings and a classification model.
"""
import os
import pickle

import numpy as np

# api/models/ — one level up from services/
_model_dir = os.path.join(os.path.dirname(__file__), "..", "models")

_model_path = os.path.join(_model_dir, "tonality_model.pk")
with open(_model_path, "rb") as _f:
    _model = pickle.load(_f)

_embedding_path = os.path.join(_model_dir, "tonality_embedding.pk")
with open(_embedding_path, "rb") as _f:
    _model_embedding = pickle.load(_f)

_le_path = os.path.join(_model_dir, "tonality_LE.pk")
with open(_le_path, "rb") as _f:
    _le = pickle.load(_f)


def _get_embeddings(text: str) -> np.ndarray:
    """Encode text into a sentence embedding vector.

    Args:
        text: Input string to encode

    Returns:
        NumPy array of embedding values
    """
    return _model_embedding.encode(text)


def tonality(text: str) -> str:
    """Predict the tonality of the given text.

    Args:
        text: Input string to classify

    Returns:
        Predicted tonality label as a string
    """
    embedding = _get_embeddings(text)
    x = np.array(embedding).reshape(1, -1)
    pred = _model.predict(x)
    pred = _le.inverse_transform(pred)
    return pred[0]
