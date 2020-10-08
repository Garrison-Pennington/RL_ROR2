from pathlib import Path


import tensorflow as tf

model = tf.keras.models.load_model(Path.expanduser(Path("~/models/ror2_yolov3_v1/")), compile=False)
model.save(Path.expanduser(Path("~/models/ror2_yolov3_v1_uncompiled/")))
