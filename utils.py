import json
import time


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
