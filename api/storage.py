"""
This file has not been fully tested, and should not be used in production.

S3 buckets were initially under consideration, though adoption was halted
and this project does not currently support S3.

This file is present in the repository for future reference.
"""
import os
import boto3
from datetime import datetime, timedelta
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import logging
import tempfile

load_dotenv()

# Configure storage type based on environment
STORAGE_TYPE = os.getenv('STORAGE_TYPE', 'local')  # 'local' or 's3'
RETENTION_DAYS = int(os.getenv('VIDEO_RETENTION_DAYS', '1'))  # Default 1 day retention

# S3 configuration
S3_BUCKET = os.getenv('S3_BUCKET')
S3_REGION = os.getenv('S3_REGION', 'eu-north-1')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Storage limits (in bytes)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB per file
MAX_DAILY_UPLOAD = 500 * 1024 * 1024  # 500MB per day
ALLOWED_EXTENSIONS = {'.mp4', '.wav', '.txt'}  # Restrict file types

# Check if we should use the Next.js public directory structure
UPLOADS_DIR = os.path.join(os.getcwd(), 'uploads')
PUBLIC_UPLOADS_PATH = os.path.join(os.getcwd(), 'public', 'uploads')

# We have multiple upload directories which might cause confusion
# Check which ones exist and prioritise the public/uploads for Next.js compatibility
if os.path.exists(PUBLIC_UPLOADS_PATH):
    UPLOADS_DIR = PUBLIC_UPLOADS_PATH
    print(f"Using Next.js public directory for uploads: {UPLOADS_DIR}")
else:
    print(f"Using standard uploads directory: {UPLOADS_DIR}")

logger = logging.getLogger(__name__)

class StorageManager:
    """Manages file storage operations with quota management"""
    def __init__(self):
        """Initialise storage backend based on configuration"""
        logger.info(f"Initialising {STORAGE_TYPE} storage")
        
        if STORAGE_TYPE == 's3':
            # S3 configuration
            try:
                self.s3 = boto3.client(
                    's3',
                    region_name=S3_REGION,
                    aws_access_key_id=AWS_ACCESS_KEY,
                    aws_secret_access_key=AWS_SECRET_KEY
                )
                # Track daily upload volume
                self.daily_upload_volume = 0
                self.last_upload_date = datetime.now().date()
            except Exception as e:
                logger.error(f"Failed to initialise S3 client: {e}")
                raise
        else:
            # Create directories without checking permissions first
            for directory in [UPLOADS_DIR, 'temp']:
                dir_path = os.path.abspath(directory)
                try:
                    os.makedirs(dir_path, mode=0o755, exist_ok=True)
                    logger.info(f"Initialised {directory} directory at {dir_path}")
                except Exception as e:
                    logger.warning(f"Could not create directory {dir_path}: {e}")
                    # Continue anyway - we'll handle specific file operations later
            
            # Note: We don't check write permissions here anymore to avoid failing at startup

    def _check_file_size(self, file_obj):
        """Check if file size is within limits"""
        # Get file size
        if hasattr(file_obj, 'seek') and hasattr(file_obj, 'tell'):
            pos = file_obj.tell()
            file_obj.seek(0, os.SEEK_END)
            size = file_obj.tell()
            file_obj.seek(pos)  # Reset position
        else:
            # For files that don't support seek/tell
            size = len(file_obj.read())
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)

        if size > MAX_FILE_SIZE:
            raise ValueError(f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB")
        return size

    def _check_extension(self, filename):
        """Check if file extension is allowed"""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"File type not allowed. Allowed types: {ALLOWED_EXTENSIONS}")
        return True

    def _update_upload_quota(self, size):
        """Update and check daily upload quota"""
        today = datetime.now().date()
        if today != self.last_upload_date:
            self.daily_upload_volume = 0
            self.last_upload_date = today

        new_total = self.daily_upload_volume + size
        if new_total > MAX_DAILY_UPLOAD:
            raise ValueError(f"Daily upload limit ({MAX_DAILY_UPLOAD/1024/1024}MB) exceeded")
        
        self.daily_upload_volume = new_total

    def save_file(self, file_obj, filename, content_type=None):
        """Save a file to storage with size and quota checks"""
        try:
            # Validate file
            self._check_extension(filename)
            size = self._check_file_size(file_obj)

            # Use system temp directory which should be writable by all users
            temp_dir = tempfile.gettempdir()  # usually /tmp on Linux
            logger.info(f"Using system temp directory: {temp_dir}")
            
            # Save to local temp first (needed for processing)
            temp_path = os.path.join(temp_dir, filename)
            logger.info(f"Saving file {filename} to temporary path: {temp_path}")
            
            if hasattr(file_obj, 'save'):
                logger.debug("Using file object's save method")
                file_obj.save(temp_path)
            else:
                logger.debug("Using manual file writing")
                file_obj.seek(0)  # Reset file position to start
                with open(temp_path, 'wb') as f:
                    content = file_obj.read()
                    logger.debug(f"Read {len(content)} bytes from file object")
                    f.write(content)
                file_obj.seek(0)  # Reset file position again for potential reuse
            
            if os.path.exists(temp_path):
                temp_size = os.path.getsize(temp_path)
                logger.debug(f"Temporary file created successfully, size: {temp_size} bytes")
            else:
                raise IOError(f"Failed to create temporary file at {temp_path}")

            if STORAGE_TYPE == 's3':
                try:
                    # Check quota
                    self._update_upload_quota(size)

                    # Upload to S3 with metadata
                    with open(temp_path, 'rb') as f:
                        extra_args = {
                            'ContentType': content_type,
                            'Metadata': {
                                'upload_date': datetime.now().isoformat(),
                                'file_size': str(size)
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
                    logger.error(f"Error uploading to S3: {e}")
                    # Don't try to move to uploads dir since we had permission issues
                    # Just return the temp path which should be accessible
                    logger.info(f"Using temp file as fallback: {temp_path}")
                    return temp_path
            else:
                # Local storage - just use the temp file path because of permission issues
                try:
                    # Try to copy to uploads directory, but don't fail if it doesn't work
                    uploads_path = os.path.join(UPLOADS_DIR, filename)
                    try:
                        # Instead of os.rename which requires permissions, try a copy
                        with open(temp_path, 'rb') as src, open(uploads_path, 'wb') as dst:
                            dst.write(src.read())
                        logger.info(f"Successfully copied file to uploads: {uploads_path}")
                        return uploads_path
                    except (PermissionError, OSError) as e:
                        logger.warning(f"Could not copy to uploads directory: {e}")
                        # Just use the temp file
                        logger.info(f"Using temp file as storage: {temp_path}")
                        return temp_path
                except Exception as e:
                    logger.warning(f"Using temp file due to error: {e}")
                    return temp_path

        except Exception as e:
            logger.error(f"Error in save_file: {e}")
            # Don't try to clean up temp file - it might be needed
            raise

    def get_file_url(self, filename, expires_in=3600):
        """Get a URL for accessing the file"""
        if not filename:
            return None
            
        if STORAGE_TYPE == 's3':
            try:
                url = self.s3.generate_presigned_url(
                    'get_object',
                    Params={
                        'Bucket': S3_BUCKET,
                        'Key': f"uploads/{filename}"
                    },
                    ExpiresIn=expires_in
                )
                return url
            except ClientError as e:
                logger.error(f"Error generating presigned URL: {e}")
                raise
        else:
            # For local development, check multiple possible locations
            
            # Check if file is in uploads directory
            uploads_path = os.path.join(UPLOADS_DIR, filename)
            if os.path.exists(uploads_path):
                logger.info(f"File {filename} found in uploads directory, returning /uploads/ URL")
                return f"/uploads/{filename}"
            
            # Check if file is in application temp directory
            app_temp_path = os.path.join('temp', filename)
            if os.path.exists(app_temp_path):
                logger.info(f"File {filename} found in application temp directory")
                return f"/temp/{filename}"
            
            # Check if file is in system temp directory
            system_temp_path = os.path.join(tempfile.gettempdir(), filename)
            if os.path.exists(system_temp_path):
                logger.info(f"File {filename} found in system temp directory")
                # For system temp files, return a /temp/ URL that will be handled by serve_temp_file
                return f"/temp/{filename}"
            
            logger.warning(f"File {filename} not found in any directory, defaulting to /uploads/ URL")
            # Default to uploads path even if not found
            return f"/uploads/{filename}"

    def delete_file(self, filename):
        """Delete a file from storage"""
        if STORAGE_TYPE == 's3':
            try:
                self.s3.delete_object(
                    Bucket=S3_BUCKET,
                    Key=f"uploads/{filename}"
                )
            except ClientError as e:
                print(f"Error deleting from S3: {e}")
                raise
        else:
            filepath = os.path.join('uploads', filename)
            if os.path.exists(filepath):
                os.remove(filepath)

    def get_file_bytes(self, filename):
        """
        Get the file contents as bytes. Works with both local and remote storage.
        
        Args:
            filename: The name of the file to retrieve
            
        Returns:
            bytes: The file contents as bytes if found
            None: If the file doesn't exist
        """
        logger.info(f"Getting file bytes for: {filename}")
        
        if STORAGE_TYPE == 's3':
            try:
                response = self.s3.get_object(
                    Bucket=S3_BUCKET,
                    Key=f"uploads/{filename}"
                )
                return response['Body'].read()
            except Exception as e:
                logger.error(f"Error retrieving file {filename} from S3: {e}")
                return None
        else:
            # For local storage, check different locations
            possible_paths = [
                os.path.join(UPLOADS_DIR, filename),              # uploads directory
                os.path.join(os.getcwd(), 'uploads', filename),   # direct uploads directory
                os.path.join(os.getcwd(), 'temp', filename),      # app temp directory
                os.path.join(tempfile.gettempdir(), filename)     # system temp directory
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"Found file at: {path}")
                    try:
                        with open(path, 'rb') as f:
                            return f.read()
                    except Exception as e:
                        logger.error(f"Error reading file {path}: {e}")
                        # Try the next path
            
            logger.warning(f"File {filename} not found in any location")
            return None

# Global storage manager instance
storage = StorageManager() 