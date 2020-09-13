import os
from io import BytesIO
from pathlib import Path

import imageio
import numpy as np
import matplotlib.pyplot as plt

vids = os.listdir(Path("C:/Users/garri/IdeaProjects/ILDataCollector/data/video/"))

with open(vids[0], "rb") as f:
    content = f.read()
    vid = imageio.get_reader(BytesIO(content), 'ffmpeg')
    for i, frame in enumerate(vid.iter_data()):
        print(frame)
