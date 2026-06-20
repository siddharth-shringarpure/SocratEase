"""
Experimental S3 and local storage manager.

This file has not been fully tested, and should not be used in production.
S3 buckets were initially under consideration, though adoption was halted
and this project does not currently support S3.
This file is present in the repository for future reference.
"""
import logging
import os
import tempfile
from datetime import datetime, timedelta

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# Configure storage type based on environment
STORAGE_TYPE = os.getenv("STORAGE_TYPE", "local")
RETENTION_DAYS = int(os.getenv("VIDEO_RETENTION_DAYS", "1"))

# S3 configuration
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION", "eu-north-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Storage limits (in bytes)
MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_DAILY_UPLOAD = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".wav", ".txt"}

UPLOADS_DIR = os.path.join(os.getcwd(), "uploads")
PUBLIC_UPLOADS_PATH = os.path.join(os.getcwd(), "public", "uploads")

# Prefer the Next.js public directory when it exists
if os.path.exists(PUBLIC_UPLOADS_PATH):
    UPLOADS_DIR = PUBLIC_UPLOADS_PATH
    print(f"Using Next.js public directory for uploads: {UPLOADS_DIR}")
else:
    print(f"Using standard uploads directory: {UPLOADS_DIR}")

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages file storage operations with quota management.

    Attributes:
        s3: Boto3 S3 client (S3 mode only)
        daily_upload_volume: Running byte count for today's uploads (S3 mode only)
        last_upload_date: Date of last recorded upload (S3 mode only)
    """

    def __init__(self) -> None:
        """Initialise storage backend based on configuration."""
        logging.info("Initialising %s storage", STORAGE_TYPE)

        if STORAGE_TYPE == "s3":
            try:
                self.s3 = boto3.client(
                    "s3",
                    region_name=S3_REGION,
                    aws_access_key_id=AWS_ACCESS_KEY,
                    aws_secret_access_key=AWS_SECRET_KEY
                )
                self.daily_upload_volume = 0
                self.last_upload_date = datetime.now().date()
            except Exception as e:
                logging.error("Failed to initialise S3 client: %s", e)
                raise
        else:
            for directory in [UPLOADS_DIR, "temp"]:
                dir_path = os.path.abspath(directory)
                try:
                    os.makedirs(dir_path, mode=0o755, exist_ok=True)
                    logging.info("Initialised %s directory at %s", directory, dir_path)
                except Exception as e:
                    logging.warning("Could not create directory %s: %s", dir_path, e)

    def _check_file_size(self, file_obj) -> int:
        """Check file size is within the allowed limit.

        Args:
            file_obj: File-like object to inspect

        Returns:
            Size of the file in bytes

        Raises:
            ValueError: If file exceeds MAX_FILE_SIZE
        """
        if hasattr(file_obj, "seek") and hasattr(file_obj, "tell"):
            pos = file_obj.tell()
            file_obj.seek(0, os.SEEK_END)
            size = file_obj.tell()
            file_obj.seek(pos)
        else:
            size = len(file_obj.read())
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)

        if size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        return size

    def _check_extension(self, filename: str) -> bool:
        """Check the file extension is permitted.

        Args:
            filename: Name of the file to validate

        Returns:
            True if the extension is allowed

        Raises:
            ValueError: If the extension is not in ALLOWED_EXTENSIONS
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File type not allowed. Allowed types: {ALLOWED_EXTENSIONS}"
            )
        return True

    def _update_upload_quota(self, size: int) -> None:
        """Update and enforce the daily upload quota.

        Args:
            size: Size in bytes of the new upload

        Raises:
            ValueError: If adding size would exceed MAX_DAILY_UPLOAD
        """
        today = datetime.now().date()
        if today != self.last_upload_date:
            self.daily_upload_volume = 0
            self.last_upload_date = today

        new_total = self.daily_upload_volume + size
        if new_total > MAX_DAILY_UPLOAD:
            raise ValueError(
                f"Daily upload limit ({MAX_DAILY_UPLOAD / 1024 / 1024}MB) exceeded"
            )

        self.daily_upload_volume = new_total

    def save_file(self, file_obj, filename: str, content_type: str | None = None) -> str:
        """Save a file to storage after validating size and extension.

        Args:
            file_obj: File-like object or Werkzeug FileStorage to save
            filename: Target filename
            content_type: MIME type of the file (optional)

        Returns:
            Path or key under which the file was saved

        Raises:
            ValueError: If validation fails
            IOError: If the temporary file cannot be created
        """
        try:
            self._check_extension(filename)
            size = self._check_file_size(file_obj)

            temp_dir = tempfile.gettempdir()
            logging.info("Using system temp directory: %s", temp_dir)

            temp_path = os.path.join(temp_dir, filename)
            logging.info("Saving file %s to temporary path: %s", filename, temp_path)

            if hasattr(file_obj, "save"):
                file_obj.save(temp_path)
            else:
                file_obj.seek(0)
                with open(temp_path, "wb") as f:
                    content = file_obj.read()
                    f.write(content)
                file_obj.seek(0)

            if not os.path.exists(temp_path):
                raise IOError(f"Failed to create temporary file at {temp_path}")

            if STORAGE_TYPE == "s3":
                try:
                    self._update_upload_quota(size)

                    with open(temp_path, "rb") as f:
                        extra_args = {
                            "ContentType": content_type,
                            "Metadata": {
                                "upload_date": datetime.now().isoformat(),
                                "file_size": str(size)
                            }
                        }
                        self.s3.upload_fileobj(
                            f,
                            S3_BUCKET,
                            f"uploads/{filename}",
                            ExtraArgs=extra_args
                        )
                    return f"uploads/{filename}"
                except Exception as e:
                    logging.error("Error uploading to S3: %s", e)
                    logging.info("Using temp file as fallback: %s", temp_path)
                    return temp_path
            else:
                uploads_path = os.path.join(UPLOADS_DIR, filename)
                try:
                    with open(temp_path, "rb") as src, open(uploads_path, "wb") as dst:
                        dst.write(src.read())
                    logging.info("Successfully copied file to uploads: %s", uploads_path)
                    return uploads_path
                except (PermissionError, OSError) as e:
                    logging.warning("Could not copy to uploads directory: %s", e)
                    logging.info("Using temp file as storage: %s", temp_path)
                    return temp_path

        except Exception as e:
            logging.error("Error in save_file: %s", e)
            raise

    def get_file_url(self, filename: str | None, expires_in: int = 3600) -> str | None:
        """Return a URL for accessing the stored file.

        Args:
            filename: Name of the file to locate
            expires_in: Presigned URL lifetime in seconds (S3 only, default: 3600)

        Returns:
            URL string if found, None if filename is empty

        Raises:
            ClientError: If generating a presigned S3 URL fails
        """
        if not filename:
            return None

        if STORAGE_TYPE == "s3":
            try:
                url = self.s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": S3_BUCKET, "Key": f"uploads/{filename}"},
                    ExpiresIn=expires_in
                )
                return url
            except ClientError as e:
                logging.error("Error generating presigned URL: %s", e)
                raise
        else:
            uploads_path = os.path.join(UPLOADS_DIR, filename)
            if os.path.exists(uploads_path):
                logging.info("File %s found in uploads directory", filename)
                return f"/uploads/{filename}"

            app_temp_path = os.path.join("temp", filename)
            if os.path.exists(app_temp_path):
                logging.info("File %s found in application temp directory", filename)
                return f"/temp/{filename}"

            system_temp_path = os.path.join(tempfile.gettempdir(), filename)
            if os.path.exists(system_temp_path):
                logging.info("File %s found in system temp directory", filename)
                return f"/temp/{filename}"

            logging.warning(
                "File %s not found in any directory, defaulting to /uploads/ URL", filename
            )
            return f"/uploads/{filename}"

    def delete_file(self, filename: str) -> None:
        """Delete a file from storage.

        Args:
            filename: Name of the file to delete

        Raises:
            ClientError: If S3 deletion fails
        """
        if STORAGE_TYPE == "s3":
            try:
                self.s3.delete_object(Bucket=S3_BUCKET, Key=f"uploads/{filename}")
            except ClientError as e:
                print(f"Error deleting from S3: {e}")
                raise
        else:
            filepath = os.path.join("uploads", filename)
            if os.path.exists(filepath):
                os.remove(filepath)

    def get_file_bytes(self, filename: str) -> bytes | None:
        """Return the file contents as bytes.

        Args:
            filename: Name of the file to retrieve

        Returns:
            File contents as bytes, or None if not found
        """
        logging.info("Getting file bytes for: %s", filename)

        if STORAGE_TYPE == "s3":
            try:
                response = self.s3.get_object(
                    Bucket=S3_BUCKET, Key=f"uploads/{filename}"
                )
                return response["Body"].read()
            except Exception as e:
                logging.error("Error retrieving file %s from S3: %s", filename, e)
                return None
        else:
            possible_paths = [
                os.path.join(UPLOADS_DIR, filename),
                os.path.join(os.getcwd(), "uploads", filename),
                os.path.join(os.getcwd(), "temp", filename),
                os.path.join(tempfile.gettempdir(), filename)
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    logging.info("Found file at: %s", path)
                    try:
                        with open(path, "rb") as f:
                            return f.read()
                    except Exception as e:
                        logging.error("Error reading file %s: %s", path, e)

            logging.warning("File %s not found in any location", filename)
            return None


# Global storage manager instance
storage = StorageManager()
