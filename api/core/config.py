"""Application configuration loaded from environment variables."""
import os


class Settings:
    """Central store for environment-driven configuration.

    Attributes:
        flask_run_port: Port for the Flask dev server (default: 5000)
    """

    flask_run_port: int = int(os.environ.get("FLASK_RUN_PORT", "5000"))


settings = Settings()
