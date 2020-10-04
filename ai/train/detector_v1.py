from os import path

import tensorflow as tf

from ai.loaders import YoloFrames
from ai.network import YOLOv3
from ai.train import yolo_loss

out_grids = ((22, 40), (45, 80), (90, 160))
training_data = YoloFrames(out_grids, 3, 1)
# t = training_data[0][1][0]
net = YOLOv3((720, 1280, 3), 17, 3, num_scales=3)
model = net.compile_model(
    loss=yolo_loss(training_data.anchors, out_grids),
    optimizer='adam'
)

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