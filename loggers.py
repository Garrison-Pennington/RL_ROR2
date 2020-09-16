from os import path, listdir
import time
from datetime import date
import atexit
import json

from winput import hook_mouse, hook_keyboard, wait_messages, stop, unhook_mouse, unhook_keyboard, winput

from utils import update_json, time_list
from control import stop_recording
from recording import start_game, start_recording

MOUSE_BUTTON_MAP = {
    513: 1, 514: 1,
    516: 2, 517: 2,
    519: 4, 520: 4,
    523: {1: 8, 2: 16}, 524: {1: 8, 2: 16},
    522: 32  # Wheel, additional winput fn for this
}  # defined by winput. See https://github.com/Zuzu-Typ/winput#sending-mouse-input
MOUSE_RELEASE_CODES = [514, 517, 520, 524]
MOUSE_PRESS_CODES = [513, 516, 519, 523]
MOUSE_WHEEL_CODE = 522
MOUSE_MOVEMENT_CODE = 512
KEY_PRESS = 256
KEY_RELEASE = 257

KEY_DIR = path.expanduser("~/IdeaProjects/ILDataCollector/data/keylogs/")
MOUSE_DIR = path.expanduser("~/IdeaProjects/ILDataCollector/data/mouselogs/")
SESSION_DIR = path.expanduser("~/IdeaProjects/ILDataCollector/data/sessions/")

start = time.time()

dt = date.today()
m = dt.month if len(str(dt.month)) == 2 else f"0{dt.month}"
d = dt.day if len(str(dt.day)) == 2 else f"0{dt.day}"
dt = f"{dt.year}{m}{d}"
del m, d

sessions_today = 0
while path.exists(KEY_DIR + dt + f"_{sessions_today}.log"):
    sessions_today += 1

key_logfile = KEY_DIR + dt + f"_{sessions_today}.log"
mouse_logfile = MOUSE_DIR + dt + f"_{sessions_today}.log"
session_file = SESSION_DIR + dt + f"_{sessions_today}.json"


def exit_handler():
    # stop recording input
    unhook_mouse()
    unhook_keyboard()
    # log end of session
    update_json(session_file, {
        "length": time.time() - start,  # Length in seconds, preserved for accuracy
        "end": time.time()
    })
    stop_recording()
    return 1


def on_keyboard_event(event):

    if event.vkCode == winput.VK_F3:
        print("stopping")
        stop()
    with open(key_logfile, 'a') as f:
        log = f"{time.time() - start} > {event.vkCode} {event.action}\n"
        f.write(log)

    return 1


def on_mouse_event(event):
    with open(mouse_logfile, 'a') as f:
        log = f"{time.time() - start}>{event.action}>{event.position}>{event.additional_data}\n"
        f.write(log)

    return 1


def start_logging(game):
    global start
    start = time.time()
    # log start time
    with open(session_file, "a") as f:
        json.dump({"input start": start}, f)
    # set the hook
    hook_keyboard(on_keyboard_event)
    hook_mouse(on_mouse_event)
    start_game(game)
    update_json(session_file, {"game start": time.time(), "game": game})
    start_recording()
    update_json(session_file, {"recording start": get_obs_log_time()})
    # set exit handler to log session data
    print("registering exit handler")
    atexit.register(exit_handler)
    # listen for input
    wait_messages()


def get_obs_log_time():
    log_dir = path.expanduser("~/AppData/Roaming/obs-studio/logs")
    files = listdir(log_dir)
    files = list(filter(lambda nm: nm[:10].replace("-", "") == dt, files))
    times = list(map(lambda nm: [int(el) for el in nm[11:21].split("-")], files))
    t = time_list()
    times = list(map(lambda tl: tl[0] - t[0] * (60 * 60) + tl[1] - t[1] * 60 + tl[2] - t[2], times))
    idx = times.index(min(times))
    rec_time = ""
    while not rec_time:
        with open(files[idx], "r") as f:
            lines = f.readlines()
            target = filter(lambda txt: "Recording Start" in txt, lines)
            if target:
                rec_time = target[:12]
            else:
                time.sleep(.5)
    ms = int(rec_time[-3:]) / 1000
    ts = time.mktime(time.strptime(f"{dt}-{rec_time[:-4]}", "%Y%m%d-%H:%M:%S")) + ms
    return ts
