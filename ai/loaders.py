import os
from io import BytesIO
from pathlib import Path
import itertools

import imageio
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

VID_DIR = Path("C:/Users/garri/IdeaProjects/ILDataCollector/data/video/")


class RawVideo(Dataset):

    @classmethod
    def load_video(cls, filepath):
        with open(filepath, "rb") as f:
            vid = imageio.get_reader(BytesIO(f.read()), 'ffmpeg')
            metadata = vid.get_meta_data()
            vid = vid.iter_data()
        return vid, metadata

    def __init__(self, video_dir):
        self.videos = os.listdir(video_dir)
        self.dir = video_dir

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        video, meta = RawVideo.load_video(os.path.join(self.dir, self.videos[item]))
        return {
            "data": video,
            "metadata": meta
        }


class Frames(Dataset):

    def __init__(self, video):
        self.video = video

    def __len__(self):
        return len([1 for _, _ in enumerate(self.video)])

    def __getitem__(self, item):
        return np.array(next(itertools.islice(self.video, item, None))).astype(np.float32) / 255
