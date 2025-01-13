import boto3
from botocore.exceptions import NoCredentialsError

S3_BUCKET_NAME = 'your-bucket-name'

def upload_to_s3(file_path, s3_key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(file_path, S3_BUCKET_NAME, s3_key)
        print(f"File {file_path} uploaded to S3 bucket {S3_BUCKET_NAME} as {s3_key}")
    except FileNotFoundError:
        print("The file was not found")
    except NoCredentialsError:
        print("Credentials not available")

