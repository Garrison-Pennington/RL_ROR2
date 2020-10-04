from tqdm import tqdm

from ai.cloud import upload, get_blob_names, TAGGED_FRAMES
from ai.loaders import WINDOWS_DETECT_DIR

bucketed = get_blob_names(TAGGED_FRAMES)

vott_assets = WINDOWS_DETECT_DIR.glob("**/*")
vott_assets = filter(lambda p: p.is_file() and p.name not in bucketed, vott_assets)

for i, p in enumerate(tqdm(vott_assets, "uploading tagged frames")):
    upload((TAGGED_FRAMES, p.name), p)
