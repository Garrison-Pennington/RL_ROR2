from os import path

import tensorflow as tf

from ai.loaders import YoloFrames
from ai.network import YOLOv3
from ai.train import yolo_loss

net = YOLOv3((720, 1280, 3), 17, 2)
model = net.compile_model(
    loss=yolo_loss(),
    optimizer='adam'
)

out_grids = [(o.shape[1], o.shape[2]) for o in model.outputs]
training_data = YoloFrames(out_grids, 2, 1)
t = training_data[0]

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path.expanduser("~/models/ror2_yolov3_v1"),
                                                 save_weights_only=True,
                                                 verbose=1)

try:
    model.fit(
        training_data,
        epochs=100,
        verbose=1,
        callbacks=[cp_callback]
    )
except KeyboardInterrupt:
    model.save(path.expanduser("~/models/ror2_yolov3_v1"))
else:
    model.save(path.expanduser("~/models/ror2_yolov3_v1"))