import json
import time
from os import path
from functools import reduce


def update_json(file, data):
    with open(file, "r+") as f:
        res = json.load(f)
    with open(file, "w") as f:
        for k in data:
            res[k] = data[k]
        json.dump(res, f)


def time_list(secs=None):
    t_inf = time.localtime() if secs is None else time.localtime(secs)
    return [t_inf[3], t_inf[4], t_inf[5]]


def time_list_from_video_name(filename):
    t = filename.split("_")[1].split(";")
    return [int(e) for e in t]


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def index_model_layers():
    with open(path.expanduser("~/model_summary.txt"), "r+") as f:
        lines = f.readlines()
    idx = 0
    for i in range(len(lines[4:])):
        if lines[i + 4][0] not in " _=":
            lines[i + 4] = f"{idx}: {lines[i+4]}"
            idx += 1
    with open(path.expanduser("~/model_summary.txt"), "w+") as f:
        f.writelines(lines)
