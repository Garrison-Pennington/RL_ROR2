import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import math
import itertools
from io import BytesIO
from pathlib import Path
from collections import defaultdict
from functools import reduce

import imageio
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf

from loggers import *
from utils import time_list
from utils.ai import logit, idx_tensor

VID_DIR = Path("C:/Users/garri/IdeaProjects/ILDataCollector/data/video/")
DETECT_DIR = Path("C:/Users/garri/data/ROR2/vott-json-export/")


class RawVideo:

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


class Frames:

    def __init__(self, video, metadata):
        self.video = video
        self.metadata = metadata

    def __len__(self):
        return self.metadata["nframes"]

    def __getitem__(self, item):
        return np.array(next(itertools.islice(self.video, item, None))).astype(np.float32) / 255


class Actions:

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
        return self.keys[item], self.mouse[item]

    def match_log_to_video(self):
        """
        Find the .mkv file corresponding to the recording session

        :return: filename of the video  Ex: 20200912_19;04;56.mkv
        """
        record_start = self.session["recording start"]
        stamp = time_list(record_start)
        best = None
        min_diff = math.inf
        names = list(filter(lambda s: s.split("_")[0] == self.session_stamp[:-2], os.listdir(VID_DIR)))
        for name in names:
            vid_ts = name[:-4].split("_")[1].split(";")
            diff = [abs(int(vid_ts[e]) - stamp[e]) for e in range(3)]
            diff = (diff[0] * (60 * 60)) + (diff[1] * 60) + diff[2]
            if diff < min_diff:
                best = name
        return best

    def key_log_to_frame_states(self, log):
        # Divide action states into frames
        offset = self.session["recording start"] - self.session["input start"]  # Delay (sec) between beginning of input log and beginning of video
        step = 1 / self.fps # time in seconds one frame fills
        active_keys = set()
        held_keys = []
        current_frame = log[0][0]
        frames = [[]] * int(self.nframes)
        for i in range(len(log)):
            frm, ev = log[i].split(">")
            frm = int((float(frm) - offset) / step)
            key, ev = [int(x) for x in ev.strip().split(" ")]
            if frm >= self.nframes:
                print("More input frames than video frames, something is wrong")
                break
            if frm > current_frame:  # Have we advanced to another frame?
                frames[current_frame] = list(active_keys)  # Record the currently pressed keys
                current_frame = frm  # Move the current frame forward
                active_keys = set(held_keys)
            if ev == KEY_PRESS:
                if key not in active_keys:
                    active_keys.add(key)
                if key not in held_keys:
                    held_keys.append(key)
            if ev == KEY_RELEASE and key in active_keys:
                held_keys.remove(key)
        return frames

    def mouse_log_to_frame_states(self, log):
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
        :return:
        """
        tier_map = [*MOUSE_PRESS_CODES, *MOUSE_RELEASE_CODES, MOUSE_WHEEL_CODE, MOUSE_MOVEMENT_CODE]
        # Divide action states into frames
        offset = self.session["recording start"] - self.session["input start"]  # Delay (sec) between beginning of input log and beginning of video
        step = 1 / self.fps  # time in seconds one frame fills
        frames = [[]] * int(self.nframes)
        for i in range(len(log)):
            time_stamp, event_code, position, additional_data = log[i].split(">")
            frame = int((float(time_stamp) - offset) / step)
            event_code = int(event_code)
            position = [int(x) for x in position[1:-1].split(",")]
            additional_data = int(additional_data[1:-3]) if additional_data[1:-3] else None
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
                groups.append(list(g_p))
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
                elif ev == 512:
                    continue
                button = MOUSE_BUTTON_MAP[ev]
                if isinstance(button, dict):  # Is it an extra button?
                    button = button[groups[i][0][-1]]
                if ev in MOUSE_PRESS_CODES:
                    active_buttons.add(button)
                    if button not in held_buttons:
                        held_buttons.append(button)
                else:
                    if button in held_buttons:
                        held_buttons.remove(button)
            frames[k] = [list(active_buttons), position, wheel_shift]
            active_buttons = set(held_buttons)
            wheel_shift = 0
        return frames


class TaggedFrames(Sequence):
    """
    Dataset of images with localized object classifications

    Data: H x W x 3 numpy array of a jpg image (float32)
    Label: Object x (x, y, w, h, o, c0-cN) numpy array (float32)
    """

    def json_to_sample(self, sample):
        asset = sample["asset"]
        data = np.asarray(*imageio.read(DETECT_DIR.joinpath(asset["name"]))).astype(np.float32) / 255
        labels = sample["regions"]
        bboxes, tags = zip(*[(b["boundingBox"], b["tags"]) for b in labels])
        bboxes = np.array([np.array([
            b["left"]+b["width"]/2,  # x
            b["top"]+b["height"]/2,  # y
            b["width"],              # w
            b["height"],             # h
            1                        # o
        ]) for b in bboxes])
        tags = list(map(lambda lot: [self.class_labels.index(t) for t in lot], tags))
        tag_arr = np.zeros((len(tags), len(self.class_labels))).astype(np.float32)
        for obj in range(len(tags)):
            for t in tags[obj]:
                tag_arr[obj][t] = 1
        label = np.concatenate((bboxes, tag_arr), axis=1)
        return data, label

    def __init__(self):
        with open(DETECT_DIR.joinpath("ROR2-Annotations-export.json")) as f:
            annotations = json.load(f)
        self.class_labels = [a["name"] for a in annotations["tags"]]
        self.samples = list(map(lambda k: self.json_to_sample(annotations["assets"][k]), annotations["assets"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]


def join_boxes(*boxes, cell_shape, image_shape):
    if len(boxes) == 1:
        (x, y, w, h, _), classes = boxes[0][:5], boxes[0][5:]
        minx, miny, maxx, maxy = x - w/2, y - h/2, x + w/2, y + h/2
        classes = np.array(classes)
    else:
        xs, ys, ws, hs, o, *classes = tuple(zip(*boxes))
        bounds = list(zip(*[(
            xs[i] - ws[i]/2,
            ys[i] - hs[i]/2,
            xs[i] + ws[i]/2,
            ys[i] + hs[i]/2
        ) for i in range(len(xs))]))
        (minx, miny), (maxx, maxy) = tuple(map(min, bounds[:2])), tuple((map(max, bounds[2:])))
        classes = list(zip(*classes))
        classes = reduce(np.logical_or, classes)
    (cell_height, cell_width), (image_height, image_width) = cell_shape, image_shape
    w, h = (maxx - minx)/image_width, (maxy - miny)/image_height
    (x, _), (y, _) = math.modf(((minx + maxx)/2)/cell_width), \
                     math.modf(((miny + maxy)/2)/cell_height)
    return np.array([x, y, w, h, 1, *classes])


class YoloFrames(Sequence):

    def __init__(self, out_grids, num_boxes, batch_size):
        self.dataset = TaggedFrames()
        self.cell_shape = (5 + len(self.dataset.class_labels)) * num_boxes
        self.out_grids = [np.zeros((*shp, self.cell_shape)) for shp in out_grids]
        self.num_boxes = num_boxes
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx):
        data, labels = list(zip(*[self.dataset[i] for i in range(idx * self.batch_size, (idx + 1) * self.batch_size)]))
        image_height, image_width, _ = data[0].shape
        labels = list(labels)
        for i in range(len(labels)):
            l = labels[i]
            label = []
            for tmp in self.out_grids:
                visited = defaultdict(list)
                arr = tmp.copy()
                y_cells, x_cells, _ = arr.shape
                cell_height, cell_width = image_height/y_cells, image_width/x_cells
                for obj in range(len(l)):
                    (x, y, w, h) = l[obj, :4]
                    (_, cx), (_, cy) = math.modf(x / cell_width), math.modf(y / cell_height)
                    visited[int(cy), int(cx)].append(l[obj])
                for k in visited:
                    tmp = join_boxes(*visited[k], cell_shape=(cell_height, cell_width), image_shape=(image_height, image_width))
                    arr[k] = np.concatenate([tmp for _ in range(self.num_boxes)])
                label.append(arr.reshape((1, *arr.shape)))
            labels[i] = label
        data = np.array(data).reshape((self.batch_size, image_height, image_width, 3))
        labels = list(zip(*labels))
        return data, labels

    # def __iter__(self):
    #     return self
    #
    # def __next__(self):
    #     if self.i == len(self):
    #         raise StopIteration
    #     item = self[self.i]
    #     self.i += 1
    #     return item