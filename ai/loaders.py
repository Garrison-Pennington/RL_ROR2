import os
import math
import itertools
from io import BytesIO
from pathlib import Path

import imageio
import numpy as np
from torch.utils.data import Dataset

from loggers import *
from utils import time_list, time_list_from_video_name

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

    def __init__(self, video, metadata):
        self.video = video
        self.metadata = metadata

    def __len__(self):
        return self.metadata["nframes"]

    def __getitem__(self, item):
        return np.array(next(itertools.islice(self.video, item, None))).astype(np.float32) / 255


class Actions(Dataset):

    def __init__(self, session_stamp):
        with open(Path(KEY_DIR + session_stamp + ".log"), "r") as f:
            self.keys = f.readlines()
        with open(Path(MOUSE_DIR + session_stamp + ".log"), "r") as f:
            self.mouse = f.readlines()
        with open(Path(SESSION_DIR + session_stamp + ".json"), "r") as f:
            self.session = json.load(f)
        self.session_stamp = session_stamp
        # Find and load the matching video metadata
        _, metadata = RawVideo.load_video(VID_DIR.joinpath(self.match_log_to_video()))
        self.nframes = math.ceil(metadata['duration'] * metadata['fps'])
        self.frames = [[]] * int(self.nframes)
        self.fps = metadata['fps']
        # Convert lines of (time, key, event) to list of Frame x key
        self.keys = self.key_log_to_frame_states(self.keys)
        self.mouse = self.mouse_log_to_frame_states(self.mouse)

    def __len__(self):
        return self.nframes

    def __getitem__(self, item):
        return 0

    def match_log_to_video(self):
        """
        Find the .mkv file corresponding to the recording session

        :return: filename of the video  Ex: 20200912_19;04;56_-7.mkv
        """
        record_start = self.session["recording start"]
        stamp = time_list(record_start)
        best = None
        min_diff = math.inf
        names = list(filter(lambda s: s.split("_")[0] == self.session_stamp[:-2], os.listdir(VID_DIR)))
        for name in names:
            vid_ts = name.split("_")[1].split(";")
            diff = [abs(int(vid_ts[e]) - stamp[e]) for e in range(3)]
            diff = (diff[0] * (60 * 60)) + (diff[1] * 60) + diff[2]
            if diff < min_diff:
                best = name
        return best

    def key_log_to_frame_states(self, log, video_filename):
        # Divide action states into frames
        rs = time_list_from_video_name(video_filename)
        rs = time.mktime(time.strptime(f"{rs[0]}:{rs[1]}:{rs[2]}", "%H:%M:%S"))
        offset = rs - self.session["input start"]  # Delay (sec) between beginning of input log and beginning of video
        step = 1 / self.fps # time in seconds one frame fills
        log = [l.split(">") for l in log]  # N x (time, event)
        # N x [time, keycode, eventcode]
        log = [[float(intm.strip()) - offset, *[int(x) for x in ev.strip().split(" ")]] for intm, ev in log]
        log = [[int(intm / step), key, ev] for intm, key, ev in log]
        active_keys = held_keys = []
        current_frame = log[0][0]
        frames = self.frames
        for frm, key, ev in log:
            if frm >= self.nframes:
                print("More input frames than video frames, something is wrong")
                break
            if frm > current_frame:  # Have we advanced to another frame?
                frames[current_frame] = active_keys  # Record the currently pressed keys
                current_frame = frm  # Move the current frame forward
                active_keys = held_keys
            if ev == KEY_PRESS:
                if key not in active_keys:
                    active_keys.append(key)
                if key not in held_keys:
                    held_keys.append(key)
            if ev == KEY_RELEASE and key in active_keys:
                held_keys.remove(key)
        return frames

    def mouse_log_to_frame_states(self, log, video_filename):
        """
        Convert a mouse input log to a list of mouse states in each video frame. The mouse state is defined as the
        mouse's position in that frame, and list of all currently pressed mouse buttons in that frame.

        There may be many recorded mouse positions for a single frame, to ensure only one is used, and important game
        actions are performed at the correct coordinates, the position saved is determined by prioritizing events and
        their positions in the following order:

        0. LMB pressed - event code = 513 = 0x0201
        1. RMB pressed - event code = 516 = 0x0204
        2. MMB pressed - event code = 519 = 0x0207
        3. XMB pressed - event code = 523 = 0x020B
        4. LMB released - event code = 514 = 0x0202
        5. RMB released - event code = 517 = 0x0205
        6. MMB released - event code = 520 = 0x0208
        7. XMB released - event code = 524 = 0x020C
        8. Avg. WHL - event code = 522 = 0x020A
        9. Avg. MOV - event code = 512 = 0x0200

        :param log:  List of lines of mouse input log
        :param video_filename:  Name of corresponding video file
        :return:
        """
        tier_map = [*MOUSE_PRESS_CODES, *MOUSE_RELEASE_CODES, MOUSE_WHEEL_CODE, MOUSE_MOVEMENT_CODE]
        # Divide action states into frames
        rs = time_list_from_video_name(video_filename)
        rs = time.mktime(time.strptime(f"{rs[0]}:{rs[1]}:{rs[2]}", "%H:%M:%S"))
        offset = rs - self.session["input start"]  # Delay (sec) between beginning of input log and beginning of video
        step = 1 / self.fps # time in seconds one frame fills
        frames = self.frames
        for i in range(len(log)):
            time_stamp, event_code, position, additional_data = log[i].split(">")
            frame = int((float(time_stamp) - offset) / step)
            event_code = int(event_code)
            position = [int(x) for x in position[1:-1].split(",")]
            additional_data = int(additional_data[1:-2]) if additional_data[1:-2] else None
            log[i] = frame, event_code, position, additional_data
        # Frame by frame, identify matching input entries and produce a frame input state
        active_buttons = set()
        held_buttons = []
        wheel_shift = 0
        frame_inputs = itertools.groupby(log, lambda inp: inp[0])
        for k, g in frame_inputs:  # k = frame #, g = input data in frame
            ev_groupby = itertools.groupby(list(g), lambda grp: grp[1])
            # Identify position tier
            groups = []
            keys = []
            for k_p, g_p in ev_groupby:  # k = ev_code, g = input data of events
                keys.append(int(k_p))
                groups.append(g_p)
            tiers = list(map(lambda e: tier_map.index(e), keys))
            grp = list(map(lambda idx: groups[idx], [i for i, x in enumerate(keys) if x == tier_map[min(tiers)]]))
            grp = [item for sub in grp for item in sub]
            # [x, y]
            position = np.mean(np.array(list(map(lambda inp: list(inp[2]), grp))), axis=0).astype(np.uint32)
            if min(tiers) == 9:
                frames[k] = [list(active_buttons), position, wheel_shift]
                continue
            # Get buttons pressed and released
            for i, ev in enumerate(keys):
                if ev == 522:  # Is it the wheel?
                    wheel_shift += sum(list(map(lambda inp: inp[-1], groups[i])))
                    continue
                button = MOUSE_BUTTON_MAP[ev]
                if isinstance(button, dict):  # Is it an extra button?
                    button = button[next(groups[i])[-1]]
                press = ev in MOUSE_PRESS_CODES
                if press:
                    active_buttons.add(button)
                    if button not in held_buttons:
                        held_buttons.append(button)
                else:
                    held_buttons.remove(button)
            frames[k] = [list(active_buttons), position, wheel_shift]
            active_buttons = set(held_buttons)
            wheel_shift = 0
        return frames
