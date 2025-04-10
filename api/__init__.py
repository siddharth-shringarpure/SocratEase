"""
API package initialisation.

This module handles basic setup tasks for the API package, including:
- Creating required temporary and upload directories
- Setting appropriate permissions for file operations
- Avoiding circular imports by keeping minimal imports
"""
from typing import List
import os
import stat
import logging

# Configure logging for directory operations
logger = logging.getLogger(__name__)

# Define required directories with proper permissions
REQUIRED_DIRS: List[str] = ['temp', 'uploads']

# Create directories needed for file operations
for directory in REQUIRED_DIRS:
    dir_path = os.path.join(os.getcwd(), directory)
    try:
        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Set full rwx permissions (777)
        os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        logger.info(f"Successfully created/updated directory: {dir_path}")
    except Exception as e:
        # Log warning but continue - application will handle file errors
        logger.warning(f"Could not set permissions on {dir_path}: {e}")

# TODO: Review if 777 permissions are too permissive for production
# TODO: Consider moving directory setup to a dedicated configuration module

if __name__ == "__main__":
    port = int(os.environ.get('FLASK_RUN_PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)