from pathlib import Path
from platform import system

import imageio
import numpy as np

from ai.cloud import local_cloud_file, get_blob_names, TAGGED_FRAMES
from ai.loaders import WINDOWS_DETECT_DIR, LINUX_DETECT_DIR


class TaggedFrame:

    def __init__(self, name):
        self.name = name
        self.dir = WINDOWS_DETECT_DIR if system() == "Windows" else LINUX_DETECT_DIR

    @property
    def image(self):
        fp = local_cloud_file((TAGGED_FRAMES, self.name + ".jpg"), self.dir.joinpath(self.name + ".jpg"))
        return np.asarray(*imageio.read(fp)).astype(np.float32) / 255

    @property
    def objects(self):
        npzfile = np.load(local_cloud_file((TAGGED_FRAMES, self.name+".tags.npz"), self.dir.joinpath(self.name + ".tags.npz")))
        return npzfile["arr_0"]


def tagged_frames():
    blob_names = get_blob_names(TAGGED_FRAMES)
    return list(
        map(lambda s: TaggedFrame(Path(s).stem),
            filter(lambda s: s != "annotations.json" and
                    not s.endswith(".tags.npz"),
                    blob_names
                   )
            ))
