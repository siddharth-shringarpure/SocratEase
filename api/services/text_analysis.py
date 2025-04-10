"""
Text analysis services for speech evaluation.

This module provides functions for analysing speech text, including:
- Filler word detection and analysis
- Type-Token Ratio (TTR) calculation for vocabulary diversity
- Logical flow scoring
"""
import re
import os
import pickle as pkl
import traceback
import torch
import transformers
from itertools import tee
from typing import Dict, List, Any, Set, Tuple, Optional

# Common filler words and phrases in English speech
FILLER_WORDS: Set[str] = {
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

def ngrams(words, n):
    """
    Generate n-grams from a list of words.

    Args:
        words: List of words to process
        n: Size of n-grams to generate

    Returns:
        List of generated n-grams as strings
    """
    output = []
    for i in range(len(words) - n + 1):
        output.append(' '.join(words[i:i + n]))
    return output

def analyse_filler_words(text: str) -> Dict[str, Any]:
    """
    Analyse text for filler words and calculate speech metrics.

    Args:
        text: The text to analyse
        
    Returns:
        Dict: containing analysis results including:
        - Word counts and percentages
        - Found filler words
        - TTR analysis
        - Logical flow score
    """
    # Extract individual words, converting to lowercase
    words: List[str] = re.findall(r'\b\w+\b', text.lower())
    total_words: int = len(words)

    # Track filler word occurrences
    filler_count: int = 0
    found_fillers: List[str] = []

    # Check for single-word fillers
    for word in words:
        if word in FILLER_WORDS:
            filler_count += 1
            found_fillers.append(word)

    # Check for multi-word filler phrases
    for n in range(2, 5):  # Check 2 to 4 word phrases
        for phrase in ngrams(words, n):
            if phrase in FILLER_WORDS:
                filler_count += 1
                found_fillers.append(phrase)

    # Calculate percentage of filler words
    filler_percentage: int = (
        round((filler_count / total_words * 100)) if total_words > 0 else 0
    )

    # Determine emoji based on filler percentage
    if filler_percentage <= 3:
        emoji = "🎯"  # excellent
    elif filler_percentage <= 7:
        emoji = "👍"  # Good
    elif filler_percentage <= 12:
        emoji = "💭"  # Think about it
    elif filler_percentage <= 18:
        emoji = "⚠️"  # Warning
    else:
        emoji = "😞"  # Needs work

    # Get additional analysis metrics
    ttr_analysis: Dict[str, Any] = calculate_ttr(text)

    # Calculate logical flow score with error handling
    try:
        logical_score: float = logical_flow(text)
        logical_percentage: float = logical_score * 100

        # Assign flow emoji based on score ranges
        if logical_percentage >= 80:
            logical_emoji = "🌠"  # Excellent flow
        elif logical_percentage >= 60:
            logical_emoji = "🔄"  # Good flow 
        elif logical_percentage >= 40:
            logical_emoji = "🔀"  # Average flow
        elif logical_percentage >= 20:
            logical_emoji = "⚠️"  # Needs work
        else:
            logical_emoji = "🚧"  # Poor flow
    except Exception as e:
        print(f"Error calculating logical flow: {str(e)}")
        logical_percentage = 0.69  # Default fallback value
        logical_emoji = "❓"  # Error indicator

    # TODO: Implement repetition detection for advanced analysis??
    # TODO: Consider adding sentiment analysis??

    return {
        "total_words": total_words,
        "filler_count": filler_count,
        "filler_percentage": filler_percentage,
        "found_fillers": found_fillers,
        "filler_emoji": emoji,
        "ttr_analysis": ttr_analysis,
        "logical_flow": {
            "score": logical_score,
            "emoji": logical_emoji
        }
    }

def calculate_ttr(text: str) -> Dict[str, Any]:
    """
    Calculate Type-Token Ratio (TTR) for vocabulary diversity analysis.
    TTR is a measure of vocabulary diversity, calculated as the ratio of unique words to total words.

    Args:
        text: Input text to analyse

    Returns:
        Dictionary containing TTR metrics and assessment
    """
    # Clean text by removing punctuation
    cleaned_text: str = re.sub(r'[^\w\s]', '', text.lower())
    words: List[str] = cleaned_text.split()

    # Calculate basic metrics
    total_words: int = len(words)
    unique_words: int = len(set(words))
    ttr: float = unique_words / total_words if total_words > 0 else 0

    # Assess vocabulary diversity level
    if ttr > 0.8:
        level, emoji = "very high", "🌟"
    elif ttr > 0.7:
        level, emoji = "high", "✨"
    elif ttr > 0.5:
        level, emoji = "average", "👍"
    elif ttr > 0.3:
        level, emoji = "low", "🔍"
    else:
        level, emoji = "very low", "📝"

    return {
        "ttr": round(ttr * 100, 2),
        "unique_words": unique_words,
        "diversity_level": level,
        "emoji": emoji
    }

def logical_flow(text: str) -> float:
    """
    Analyse logical flow and coherence of text using ML model.

    Args:
        text: Input text to analyse

    Returns:
        Float score between 0 and 1 indicating logical flow quality

    Raises:
        Various exceptions during model loading/prediction
    """
    try:
        model_path: str = os.path.join(
            os.path.dirname(__file__), 'model', 'logical_model.pk'
        )
        print(f"Loading model from: {model_path}")

        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Model not found at: {model_path}")
            return 0.69

        # Log version information for debugging
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")

        # Load and validate model
        with open(model_path, 'rb') as f:
            try:
                logical_model = pkl.load(f)
                print("Successfully loaded logical flow model")
            except RuntimeError as e:
                if "register_pytree_node()" in str(e):
                    print("Version mismatch detected between PyTorch and transformers")
                    print("Please ensure compatible versions are installed")
                    return 0.0069
                raise

        # Make prediction
        print(f"Processing text of length: {len(text)}")
        pred: List[Dict[str, float]] = logical_model.predict(text)
        score: float = pred[0]['score']
        print(f"Flow score: {score}")

        return score

    except Exception as e:
        print(f"Error in flow analysis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return 0.69