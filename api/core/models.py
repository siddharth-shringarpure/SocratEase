"""Shared request and response models."""
from dataclasses import dataclass, field


@dataclass
class TTSRequest:
    """Body for TTS generation endpoints.

    Attributes:
        text: Text to synthesise
        voice: Voice style name, eg: "M1", "F2" (default: service default)
        speed: Speech speed, 0.7--2.0 (default: 1.05)
        category: Practice category forwarded as a response header
    """

    text: str
    voice: str | None = None
    speed: float = 1.05
    category: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "TTSRequest":
        """Build a TTSRequest from a parsed JSON dict.

        Args:
            data: Raw request payload

        Returns:
            Populated TTSRequest instance
        """
        return cls(
            text=data.get("text", ""),
            voice=data.get("voice"),
            speed=data.get("speed", 1.05),
            category=data.get("category"),
        )
