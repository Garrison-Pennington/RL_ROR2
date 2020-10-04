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
        fp = local_cloud_file((TAGGED_FRAMES, self.dir.joinpath(self.name)))
        return np.asarray(*imageio.read(fp)).astype(np.float32) / 255

    @property
    def objects(self):
        fp = local_cloud_file((TAGGED_FRAMES, "annotations.json"), self.dir.joinpath("annotations.json"))


def tagged_frames():
    return map(lambda s: TaggedFrame(s),
               filter(lambda s: s != "annotations.json",
                      get_blob_names(TaggedFrame)
                      )
               )
