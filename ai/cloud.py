from pathlib import Path

from google.cloud import storage

GAME_VIDEOS = "game_videos"
INPUT_LOGS = "input_logs"

storage_client = storage.Client()


def local_cloud_file(cloud_filename):
    bucket, blob = cloud_filename
    local_path = Path.expanduser(Path(f"~/{bucket}/{blob}"))
    if Path.exists(local_path):
        return local_path
    return download(cloud_filename)


def download(cloud_filename, local_filename=None):
    bucket_name, blob_name = cloud_filename
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    local_filename = Path.expanduser(Path(f"~/{bucket_name}/{blob_name}")) if local_filename is None else local_filename
    blob.download_to_filename(local_filename)
    return local_filename


def upload(cloud_filename, local_filename):
    bucket_name, blob_name = cloud_filename
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_filename)