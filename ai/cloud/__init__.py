from pathlib import Path

from google.cloud import storage

GAME_VIDEOS = "game_videos"
INPUT_LOGS = "input_logs"
TAGGED_FRAMES = "tagged_frames"

storage_client = storage.Client()


def local_cloud_file(cloud_filename, local_filename=None):
    bucket, blob = cloud_filename

    if local_filename is not None:
        if Path.exists(local_filename):
            return local_filename
    else:
        local_filename = Path.expanduser(Path(f"~/tmp/{bucket}/{blob}"))

    try:
        Path.mkdir(local_filename.parent, parents=True, exist_ok=True)
    except FileExistsError:
        pass

    return download(cloud_filename, local_filename)


def download(cloud_filename, local_filename):
    bucket_name, blob_name = cloud_filename
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_filename)
    return local_filename


def upload(cloud_filename, local_filename):
    bucket_name, blob_name = cloud_filename
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_filename)


def get_blob_names(bucket):
    return list(map(lambda b: b.name, storage_client.list_blobs(bucket)))