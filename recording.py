import subprocess
import os
from pathlib import Path

GAME_IDS = {
    "Risk of Rain 2": 632360,
}

OBS_DIR = Path("C:/Program Files/obs-studio/bin/64bit/")
STEAM_DIR = Path("D:/Games/Steam/")


def start_recording():
    os.chdir(OBS_DIR)
    subprocess.Popen(["obs64.exe", "--startrecording", "--always-on-top"], creationflags=subprocess.CREATE_NEW_CONSOLE)


def start_game(game_name):
    os.chdir(STEAM_DIR)
    subprocess.Popen(["steam.exe", f"steam://rungameid/{GAME_IDS[game_name]}"], creationflags=subprocess.CREATE_NEW_CONSOLE)
