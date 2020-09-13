from os import path
import time
from datetime import date
import atexit
import json

from winput import hook_mouse, hook_keyboard, wait_messages, stop, unhook_mouse, unhook_keyboard, winput

from utils import update_json
from control import stop_recording
from recording import start_game, start_recording


KEY_DIR = path.expanduser("~/IdeaProjects/ILDataCollector/data/keylogs/")
MOUSE_DIR = path.expanduser("~/IdeaProjects/ILDataCollector/data/mouselogs/")
SESSION_DIR = path.expanduser("~/IdeaProjects/ILDataCollector/data/sessions/")

start = time.time()

dt = date.today()
dt = f"{dt.year}{dt.month}{dt.day}"

i = 0
while path.exists(KEY_DIR + dt + f"_{i}.log"):
    i += 1

key_logfile = KEY_DIR + dt + f"_{i}.log"
mouse_logfile = MOUSE_DIR + dt + f"_{i}.log"
session_file = SESSION_DIR + dt + f"_{i}.json"


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


def start_logging():
    global start
    start = time.time()
    # log start time
    with open(session_file, "a") as f:
        json.dump({"input start": start}, f)
    # set the hook
    hook_keyboard(on_keyboard_event)
    hook_mouse(on_mouse_event)
    start_game("Risk of Rain 2")
    update_json(session_file, {"game start": time.time()})
    start_recording()
    update_json(session_file, {"recording start": time.time()})
    # set exit handler to log session data
    print("registering exit handler")
    atexit.register(exit_handler)
    # listen for input
    wait_messages()


start_logging()
