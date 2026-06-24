"""Application configuration loaded from environment variables."""
import os


class Settings:
    """Central store for environment-driven configuration.

    Attributes:
        app_port: Port for the backend server (default: 5000)
    """

    app_port: int = int(os.environ.get("APP_PORT", "5000"))


settings = Settings()
