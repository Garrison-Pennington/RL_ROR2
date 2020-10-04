import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import math
import itertools
import json
from io import BytesIO
from pathlib import Path
from collections import defaultdict
from functools import reduce

import imageio
import numpy as np
from tensorflow.keras.utils import Sequence
import tensorflow as tf

from loggers import (
    KEY_DIR,
    MOUSE_DIR,
    SESSION_DIR,
    KEY_PRESS,
    KEY_RELEASE,
    MOUSE_WHEEL_CODE,
    MOUSE_PRESS_CODES,
    MOUSE_MOVEMENT_CODE,
    MOUSE_RELEASE_CODES,
    MOUSE_BUTTON_MAP,
)
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

    Data: batch(1) x H x W x 3 numpy array of a jpg image (float32)
    Label: batch(1) x Object x (x, y, w, h, o, c0-cN) numpy array (float32)
    """

    def json_to_sample(self, sample):
        asset = sample["asset"]
        data = np.asarray(*imageio.read(DETECT_DIR.joinpath(asset["name"]))).astype(np.float32) / 255
        data = data.reshape((1, *data.shape))
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
        return data, label.reshape((1, *label.shape))

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
    """
    Dataset of images/labels in YOLO prediction format

    Data: batch_size, 720, 1280, 3 RGB image
    Label: Complicated
        list of len(out_grids):
            l[i] = np.ndarray;
                shape: batch_size, *out_grids[i], (5 + C) * num_boxes

    indexing:
        dataset
            [i-th batch]
            [image or label]
        (image) [i-th in batch]
                [row]
                [col]
                [rgb]
        (label) [i-th out_grid]
                [i-th in batch]
                [grid_row]
                [grid_col]
                [xywho + classes]
    """

    def __init__(self, out_grids, num_boxes, batch_size):
        self.dataset = TaggedFrames()
        self.batch_size = batch_size
        self.data, labels = list(zip(*self.dataset))
        self.anchors = np.split(get_anchors(self.dataset, num_boxes*len(out_grids)), len(out_grids), axis=0)
        _, *image_shape = self.data[0].shape
        self.data = [np.vstack(self.data[i*batch_size:(i+1)*batch_size]) for i in range(len(self))]
        labels = [[bbox_to_yolo(l.copy(), self.anchors[i], out_grids[-i], image_shape)
                        for i in range(len(out_grids))] for l in labels]
        self.labels = [labels[i*batch_size:(i+1)*batch_size] for i in range(len(self))]
        self.labels = [[np.vstack(l) for l in list(zip(*batch))] for batch in self.labels]
        for batch in self.labels:
            batch.sort(key=lambda arr: arr.shape[1])

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def bbox_to_yolo(y, anchors, out_hw, img_shape):
    """

    :param y: batch x N x (5 + num_classes)
    :param anchors: num_boxes x 2
    :param out_hw: (int, int) H x W elements of returned array
    :param img_shape: (int, int) H x W of source image
    :return: batch x H x W x (num_boxes * (5 + num_classes))
    """
    num_boxes = len(anchors)
    out_params = y.shape[-1] * num_boxes
    im_h, im_w, *_ = img_shape
    grid = np.zeros((*out_hw, num_boxes, y.shape[-1]), dtype=np.float32)
    ch, cw = im_h / out_hw[0], im_w / out_hw[1]
    y[..., :4] /= [cw, ch, im_w, im_h]
    xy, cr = np.modf(y[..., :2])
    # when 2+ objects in the same cell match the same prior, assign the larger object and shift the other
    crb = np.concatenate((cr, match_to_anchors(y[..., :4], anchors).reshape((1, -1, 1))), axis=-1).astype(np.uint8)  # batch x N x 3
    _, dup_idx = np.unique(crb, axis=-1, return_inverse=True)  # (batch_idx, n_idx)
    if np.ndarray(dup_idx).any():
        print("2 objects matched the same cell prior")
        crb = shift_duplicates(crb, dup_idx, num_boxes, y)
    c, r, b = crb[..., 0], crb[..., 1], crb[..., 2]
    xy[xy == 0] = 1/max(cw, ch)
    y[..., :2] = np.log(xy/(1-xy))
    y[..., 2:4] /= anchors[b]
    y[..., 2:4] = np.log(y[..., 2:4])
    grid[r, c, b] = y
    return grid.reshape((1, *out_hw, out_params))


def shift_duplicates(crb, dup_idx, num_boxes, y):
    dup_crb = crb[..., dup_idx]
    areas = np.product(y[..., 2:4], axis=-1)[..., dup_idx]  # num_dupes
    l_idx = np.argsort(areas)
    dup_crb, dni = dup_crb[l_idx], dup_idx[l_idx]
    seen = defaultdict(list)
    for i, (c, r, b) in enumerate(dup_crb):
        k = f"{c}{r}{b}"
        if b in seen[k]:
            choices = [i for i in range(num_boxes)]
            choices.remove(b)
            while b in seen[k]:
                b = choices.pop()
            seen[k].append(b)
            crb[0, dni[i], 2] = b
        else:
            seen[k].append(b)
    return crb


def match_to_anchors(xywh, anchors):
    """

    :param xywh: batch x N x 4
    :param anchors: num_boxes x 2
    :return: N indexes in range [0, len(anchors))
    """
    num_boxes = len(anchors)
    xy, wh = xywh[..., :2], xywh[..., 2:4]  # batch x N x 2, batch x N x 2
    xy = np.tile(xy - wh/2, (num_boxes, 1, 1, 1)).transpose((1, 2, 0, 3))  # batch x N x num_boxes x 2
    wh = np.tile(wh, (num_boxes, 1, 1, 1)).transpose((1, 2, 0, 3))  # batch x N x num_boxes x 2
    centers = np.full(xy.shape, .5, dtype=np.float32) - anchors/2
    axy = np.add(centers, np.modf(xy)[1])  # batch x N x num_boxes x 2
    rb, arb = xy + wh, axy + anchors  # batch x N x num_boxes x 2
    intersection = np.product(np.fmin(rb, arb) - np.fmax(xy, axy), axis=-1)  # batch x N x num_boxes
    anchor_area = np.product(anchors, axis=-1)  # num_boxes
    label_area = np.product(wh, axis=-1)  # batch x N x num_boxes
    union = np.subtract(np.add(anchor_area, label_area), intersection)  # batch x N x num_boxes
    iou = intersection / union  # batch x N x num_boxes
    return np.argmax(iou, axis=-1)  # batch x N


def get_anchors(frames, num_clusters):

    imgs, labels = zip(*frames)
    _, im_height, im_width, *_ = imgs[0].shape
    labels = np.concatenate(labels, axis=1)
    wh_data = labels.reshape((labels.shape[1], labels.shape[2]))[..., 2:4] / [im_width, im_height]

    def input_fn():
        return tf.compat.v1.train.limit_epochs(
            tf.convert_to_tensor(wh_data, dtype=tf.float32), num_epochs=1)

    kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters, use_mini_batch=False)

    iters = 10
    for _ in range(iters):
        kmeans.train(input_fn)
        centers = kmeans.cluster_centers()

    return centers[np.argsort(centers[:, 0] * centers[:, 1])]
