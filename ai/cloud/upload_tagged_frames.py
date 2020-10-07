import json
from pathlib import Path

from tqdm import tqdm
import numpy as np

from ai.cloud import upload, get_blob_names, TAGGED_FRAMES
from ai.loaders import WINDOWS_DETECT_DIR, TaggedFrames


def save_tags(filepath, label):
    np.savez_compressed(filepath, label)
    return Path(str(filepath) + ".npz")


bucketed = get_blob_names(TAGGED_FRAMES)

vott_assets = list(WINDOWS_DETECT_DIR.glob("**/*"))
vott_assets = list(filter(lambda p: p.is_file() and p.name not in bucketed, vott_assets))
annotations = list(filter(lambda p: p.name == "annotations.json", vott_assets))[0]
vott_assets = list(filter(lambda p: p.suffix != ".npz" and p.stem != "annotations", vott_assets))

with open(annotations, "r") as f:
    annotations = json.load(f)

class_labels = list(map(lambda tag: tag["name"], annotations["tags"]))
annotations = list(
    map(lambda asset: {
        Path(asset["asset"]["name"]).stem: save_tags(
            WINDOWS_DETECT_DIR.joinpath(Path(asset["asset"]["name"]).stem + ".tags"),
            TaggedFrames.regions_to_label(
                asset["regions"],
                class_labels
            )
        )
    }, map(lambda k:
           annotations["assets"][k],
           annotations["assets"])
        )
)

tags = {}
for d in annotations:
    tags.update(d)

for i, p in enumerate(tqdm(vott_assets, "uploading tagged frames")):
    upload((TAGGED_FRAMES, p.name), p)
    upload((TAGGED_FRAMES, f"{p.stem}.tags.npz"), tags[p.stem])
