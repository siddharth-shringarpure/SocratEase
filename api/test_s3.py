"""
Untested and deprecated test script for verifying S3 connectivity and operations.

This module provides functionality to test AWS S3 connectivity, upload capabilities,
presigned URL generation, and cleanup operations. It's primarily used for
validating S3 configuration and credentials.
"""

import boto3
import io
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def test_s3_connection() -> None:
    """
    Test S3 connection and perform basic operations.

    Performs a series of tests including:
    - Connection to S3 bucket
    - File upload
    - Presigned URL generation
    - File deletion

    Raises:
        Exception: If any S3 operation fails
    """
    # Print test initialisation message
    print("Testing S3 connection...")
    
    # Display configuration details, excluding sensitive information
    print(f"\nConfiguration:")
    print(f"S3 Bucket: {os.getenv('S3_BUCKET')}")
    print(f"S3 Region: {os.getenv('S3_REGION')}")
    print(f"Storage Type: {os.getenv('STORAGE_TYPE')}")
    
    try:
        # Initialise S3 client with credentials from environment
        s3_client = boto3.client(
            's3',
            region_name=os.getenv('S3_REGION'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Verify bucket accessibility
        print("\nTesting bucket access...")
        s3_client.list_objects_v2(
            Bucket=os.getenv('S3_BUCKET', ''),  # type: ignore
            MaxKeys=1
        )
        print("✅ Successfully connected to S3 bucket")
        
        # Create and upload test file
        print("\nCreating test file...")
        test_content = b"Hello, this is a test file"
        test_file = io.BytesIO(test_content)
        
        print("Uploading to S3...")
        s3_client.upload_fileobj(
            test_file,
            os.getenv('S3_BUCKET', ''),  # type: ignore
            'test.txt'
        )
        print("✅ Upload successful")
        
        # Generate temporary access URL
        print("\nGenerating presigned URL...")
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': os.getenv('S3_BUCKET', ''),  # type: ignore
                'Key': 'test.txt'
            },
            ExpiresIn=3600  # URL valid for 1 hour
        )
        print(f"✅ URL generated: {url}")
        
        # Clean up test file
        print("\nDeleting test file...")
        s3_client.delete_object(
            Bucket=os.getenv('S3_BUCKET', ''),  # type: ignore
            Key='test.txt'
        )
        print("✅ Delete successful")
        
        print("\n🎉 All S3 operations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease check your S3 configuration and credentials.")


if __name__ == "__main__":
    # TODO: Add command line arguments for custom bucket and region testing
    test_s3_connection()