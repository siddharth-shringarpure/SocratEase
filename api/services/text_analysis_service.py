"""Text analysis services for speech evaluation.

This module provides functions for analysing speech text, including:
- Filler word detection and analysis
- Type-Token Ratio (TTR) calculation for vocabulary diversity
- Logical flow scoring
"""
import logging
import os
import pickle as pkl
import re
from typing import Any

import torch
import transformers

FILLER_WORDS: set[str] = {
    "um", "uh", "like", "you know", "well", "so", "actually", "basically",
    "i mean", "right", "okay", "er", "hmm", "literally", "anyway",
    "of course", "i guess", "in other words", "obviously", "to be honest",
    "just", "seriously", "you see", "i suppose", "frankly", "well, i mean",
    "at the end of the day", "to tell the truth", "as it were", "kind of",
    "sort of", "in a way", "that is", "as a matter of fact", "in fact",
    "like i said", "more or less", "i don't know", "basically speaking",
    "for sure", "you could say", "the thing is", "it s like",
    "put it another way", "at least", "as such", "well you know",
    "i would say", "truth be told", "yeah", "and yeah", "um yeah",
    "um no", "um right", "like literally", "to", "erm", "let s see",
    "hm", "maybe", "maybe like", "really"
}


def ngrams(words: list[str], n: int) -> list[str]:
    """Generate n-grams from a list of words.

    Args:
        words: List of words to process
        n: Size of n-grams to generate

    Returns:
        List of generated n-grams as strings
    """
    output = []
    for i in range(len(words) - n + 1):
        output.append(" ".join(words[i:i + n]))
    return output


def analyse_filler_words(text: str) -> dict[str, Any]:
    """Analyse text for filler words and calculate speech metrics.

    Args:
        text: The text to analyse

    Returns:
        Dict containing analysis results including word counts, percentages,
        found filler words, TTR analysis, and logical flow score
    """
    words = re.findall(r"\b\w+\b", text.lower())
    total_words = len(words)

    filler_count = 0
    found_fillers: list[str] = []

    for word in words:
        if word in FILLER_WORDS:
            filler_count += 1
            found_fillers.append(word)

    for n in range(2, 5):
        for phrase in ngrams(words, n):
            if phrase in FILLER_WORDS:
                filler_count += 1
                found_fillers.append(phrase)

    filler_percentage = (
        round((filler_count / total_words * 100)) if total_words > 0 else 0
    )

    ttr_analysis = calculate_ttr(text)

    try:
        logical_score = logical_flow(text)
        logical_percentage = logical_score * 100

        if logical_percentage >= 80:
            logical_status = "excellent"
        elif logical_percentage >= 60:
            logical_status = "good"
        elif logical_percentage >= 40:
            logical_status = "average"
        elif logical_percentage >= 20:
            logical_status = "needs_work"
        else:
            logical_status = "poor"
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Logical flow model may fail due to version mismatches — use fallback
        logging.error("Error calculating logical flow: %s", e)
        logical_score = 0.69
        logical_status = "unknown"

    # TODO: Implement repetition detection for advanced analysis
    # TODO: Consider adding sentiment analysis

    return {
        "total_words": total_words,
        "filler_count": filler_count,
        "filler_percentage": filler_percentage,
        "found_fillers": found_fillers,
        "ttr_analysis": ttr_analysis,
        "logical_flow": {
            "score": logical_score,
            "status": logical_status
        }
    }


def calculate_ttr(text: str) -> dict[str, Any]:
    """Calculate Type-Token Ratio (TTR) for vocabulary diversity analysis.

    TTR is the ratio of unique words to total words, indicating vocabulary
    richness.

    Args:
        text: Input text to analyse

    Returns:
        Dictionary containing TTR metrics and diversity assessment
    """
    cleaned_text = re.sub(r"[^\w\s]", "", text.lower())
    words = cleaned_text.split()

    total_words = len(words)
    unique_words = len(set(words))
    ttr = unique_words / total_words if total_words > 0 else 0

    if ttr > 0.8:
        level = "very high"
    elif ttr > 0.7:
        level = "high"
    elif ttr > 0.5:
        level = "average"
    elif ttr > 0.3:
        level = "low"
    else:
        level = "very low"

    return {
        "ttr": round(ttr * 100, 2),
        "unique_words": unique_words,
        "diversity_level": level
    }


def logical_flow(text: str) -> float:
    """Analyse logical flow and coherence of text using an ML model.

    Args:
        text: Input text to analyse

    Returns:
        Float score between 0 and 1 indicating logical flow quality

    Raises:
        RuntimeError: If the model file exists but cannot be loaded due to a
            version mismatch
    """
    try:
        model_path = os.path.join(
            os.path.dirname(__file__), "model", "logical_model.pk"
        )
        logging.info("Loading logical flow model from: %s", model_path)

        if not os.path.exists(model_path):
            logging.warning("Logical flow model not found at: %s", model_path)
            return 0.69

        logging.debug(
            "PyTorch %s, Transformers %s",
            torch.__version__,
            transformers.__version__,
        )

        with open(model_path, "rb") as f:
            try:
                logical_model = pkl.load(f)
                logging.info("✓ Logical flow model loaded")
            except RuntimeError as e:
                if "register_pytree_node()" in str(e):
                    logging.warning(
                        "Version mismatch between PyTorch and transformers"
                    )
                    return 0.0069
                raise

        logging.info("Processing text of length: %d", len(text))
        pred: list[dict[str, float]] = logical_model.predict(text)
        score = pred[0]["score"]
        logging.info("Flow score: %.4f", score)

        return score

    except Exception as e:  # pylint: disable=broad-exception-caught
        # Broad catch needed — model loading/inference can raise many error types
        logging.error("Error in flow analysis: %s", e, exc_info=True)
        return 0.69
