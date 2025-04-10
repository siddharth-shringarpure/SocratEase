"""
Templates and generators for speech feedback.

This module provides templates and functions for generating personalised feedback
on speech practice sessions, considering various speaking categories and metrics.
"""
from typing import Dict, Optional, Any

# Default response when analysis fails
DEFAULT_FEEDBACK_TEXT = (
    "Thank you for your speech. Whilst we couldn't analyse the specific content, "
    "keep practising your speaking skills regularly. Good communication is a "
    "valuable skill that improves with consistent practice."
)

# Templates for different practice categories
# TODO: Add more specific feedback templates for additional categories
CATEGORY_FEEDBACK_TEMPLATES = {
    "persuasive": (
        "Persuasive speaking is about building compelling arguments. Focus on "
        "clear points, supporting evidence, and addressing counterarguments."
    ),
    "emotive": (
        "Emotive speaking connects through feelings. Work on using emotional "
        "language, varying your tone, and building rapport with your audience."
    ),
    "public-speaking": (
        "Public speaking requires confidence and structure. Practice your "
        "opening, maintain eye contact, and organise your content with clear "
        "transitions."
    ),
    "rizzing": (
        "Charismatic speaking is about authenticity and connection. Focus on "
        "active listening, natural flow, and genuine engagement."
    ),
    "basic-conversations": (
        "Casual conversation skills build from practice. Try asking open-ended "
        "questions, maintaining a natural pace, and actively listening."
    ),
    "formal-conversations": (
        "Formal communication requires precision and clarity. Work on concise "
        "phrasing, appropriate terminology, and professional tone."
    ),
    "debating": (
        "Effective debate combines logical arguments with quick thinking. "
        "Practice outlining your key points, anticipating counterarguments, "
        "and delivering with confidence."
    ),
    "storytelling": (
        "Storytelling creates connection through narrative. Focus on your "
        "story's structure, descriptive language, and engaging delivery."
    )
}


def generate_feedback_text(
    analysis: Optional[Dict[str, Any]] = None,
    practice_category: Optional[str] = None,
    default_text: str = DEFAULT_FEEDBACK_TEXT
) -> str:
    """
    Generate personalised feedback based on speech analysis and category.

    Args:
        analysis: Dictionary containing speech analysis metrics
        practice_category: The type of speech practice being evaluated
        default_text: Fallback text when analysis is unavailable

    Returns:
        A formatted feedback string with personalised recommendations
    """
    # Return category-specific default if no analysis but category exists
    if not analysis and practice_category in CATEGORY_FEEDBACK_TEMPLATES:
        return (
            f"Here's some feedback on your {practice_category} practice.\n\n"
            f"{CATEGORY_FEEDBACK_TEMPLATES[practice_category]}\n\n"
            "While I couldn't analyse the specific content of your speech, "
            "remember that regular practice is key to improvement.\n\n"
            f"Keep practising to enhance your {practice_category} skills!"
        )

    # Return generic feedback if no analysis or category
    if not analysis:
        return default_text

    # Generate filler word feedback based on percentage
    filler_percentage = analysis["filler_percentage"]
    if filler_percentage <= 3:
        filler_comment = (
            "You used very few filler words, which makes your speech sound "
            "polished and confident."
        )
    elif filler_percentage <= 7:
        filler_comment = (
            "You used a good amount of filler words. Continuing to reduce them "
            "will make your delivery smoother."
        )
    elif filler_percentage <= 12:
        filler_comment = (
            "Try to reduce your use of filler words like 'um' and 'uh' to "
            "sound more confident."
        )
    elif filler_percentage <= 18:
        filler_comment = (
            "Your speech contained quite a few filler words, which can "
            "distract listeners from your message."
        )
    else:
        filler_comment = (
            "Work on reducing filler words significantly to improve the "
            "clarity and impact of your speech."
        )

    # Assess vocabulary diversity
    ttr_level = analysis["ttr_analysis"]["diversity_level"]
    if ttr_level == "very high":
        diversity_comment = (
            "Your vocabulary was very diverse, which made your speech "
            "engaging and precise."
        )
    elif ttr_level == "high":
        diversity_comment = (
            "Your vocabulary was varied and expressive, which strengthens "
            "your communication."
        )
    elif ttr_level == "average":
        diversity_comment = (
            "Your vocabulary diversity was good, with room to incorporate "
            "more varied terms."
        )
    elif ttr_level == "low":
        diversity_comment = (
            "Consider expanding your vocabulary to make your speech more "
            "engaging."
        )
    else:
        diversity_comment = (
            "Try to use a wider range of words to enhance your speech's impact."
        )

    # Evaluate logical flow
    flow_score = analysis["logical_flow"]["score"]
    if flow_score >= 80:
        flow_comment = (
            "Your ideas flowed together excellently, creating a cohesive "
            "narrative."
        )
    elif flow_score >= 60:
        flow_comment = "Your speech had good logical progression between points."
    elif flow_score >= 40:
        flow_comment = (
            "The logical flow was adequate, but could use stronger transitions "
            "between ideas."
        )
    elif flow_score >= 20:
        flow_comment = (
            "Work on strengthening the connections between your points for "
            "better flow."
        )
    else:
        flow_comment = "Focus on organising your thoughts more logically when speaking."

    # Generate category-specific advice
    category_specific_advice = ""
    if practice_category in CATEGORY_FEEDBACK_TEMPLATES:
        if practice_category == "persuasive":
            if flow_score < 60:
                category_specific_advice = (
                    "For persuasive speaking, try to strengthen your logical "
                    "flow with clearer transitions between arguments."
                )
            elif analysis["filler_percentage"] > 10:
                category_specific_advice = (
                    "When persuading others, reducing filler words can make "
                    "your arguments sound more authoritative."
                )
            else:
                category_specific_advice = (
                    "Your persuasive speech is developing well. Continue using "
                    "evidence and addressing counterarguments."
                )
        elif practice_category == "emotive":
            if ttr_level in ["low", "very low"]:
                category_specific_advice = (
                    "For emotive speaking, try using more varied emotional "
                    "vocabulary to express your feelings with greater nuance."
                )
            else:
                category_specific_advice = (
                    "Your emotive speech conveys feelings well. Keep matching "
                    "your tone to the emotions you're expressing."
                )
        elif practice_category == "public-speaking":
            if analysis["filler_percentage"] > 7:
                category_specific_advice = (
                    "For public speaking, work on reducing filler words to "
                    "sound more polished and confident on stage."
                )
            elif flow_score < 60:
                category_specific_advice = (
                    "Public speaking benefits from clear structure. Try "
                    "outlining your main points more clearly."
                )
            else:
                category_specific_advice = (
                    "Your public speaking skills are developing well. Keep "
                    "focusing on clear projection and structure."
                )
        elif practice_category == "rizzing":
            category_specific_advice = (
                "In charismatic conversation, your natural flow is important. "
                "Stay authentic while working on smooth transitions."
            )
        elif practice_category == "basic-conversations":
            if analysis["filler_percentage"] > 15:
                category_specific_advice = (
                    "Even in casual conversation, reducing filler words can "
                    "help you sound more articulate."
                )
            else:
                category_specific_advice = (
                    "Your casual conversation style is progressing well. "
                    "Continue asking open-ended questions."
                )
        elif practice_category == "formal-conversations":
            if ttr_level in ["low", "very low"]:
                category_specific_advice = (
                    "In formal settings, a more diverse vocabulary can "
                    "enhance your professionalism."
                )
            else:
                category_specific_advice = (
                    "Your formal communication style is developing "
                    "appropriately. Maintain your concise and clear approach."
                )
        elif practice_category == "debating":
            if flow_score < 70:
                category_specific_advice = (
                    "Debate requires strong logical flow. Focus on connecting "
                    "your arguments more clearly."
                )
            else:
                category_specific_advice = (
                    "Your debate skills show good logical reasoning. Continue "
                    "supporting claims with evidence."
                )
        elif practice_category == "storytelling":
            if ttr_level in ["low", "very low"]:
                category_specific_advice = (
                    "Storytelling benefits from rich, descriptive language. "
                    "Try expanding your vocabulary."
                )
            elif flow_score < 60:
                category_specific_advice = (
                    "Stories need a clear narrative arc. Work on transitions "
                    "between parts of your story."
                )
            else:
                category_specific_advice = (
                    "Your storytelling is developing well. Keep focusing on "
                    "narrative structure and audience engagement."
                )
    else:
        category_specific_advice = (
            "Continue practising your speaking skills regularly to improve "
            "over time."
        )

    # Construct final feedback
    feedback_text = (
        f"Here's a summary of your {practice_category or 'speech'} practice "
        f"analysis.\n\n{filler_comment} {diversity_comment} {flow_comment}\n\n"
        f"{category_specific_advice}\n\nYou said {analysis['total_words']} "
        f"total words, and about {analysis['filler_percentage']} percent of "
        f"them were filler words.\n\nKeep practising to improve your "
        f"{practice_category or 'speaking'} skills!"
    )

    return feedback_text