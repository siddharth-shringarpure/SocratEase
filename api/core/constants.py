"""Shared application constants."""

DEFAULT_FEEDBACK_TEXT = (
    "Thank you for your speech. Whilst we couldn't analyse the specific content, "
    "keep practising your speaking skills regularly. Good communication is a "
    "valuable skill that improves with consistent practice."
)

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
    ),
}
