from pathlib import Path

import tensorflow as tf

from ai.loaders import YoloFrames
from ai.network import YOLOv3
from ai.train import yolo_loss
from ai.cloud.label import tagged_frames

out_grids = ((11, 20), (22, 40), (45, 80))
training_data = YoloFrames(tagged_frames(), out_grids, 3, 1)
# t = training_data[0][1][0]
net = YOLOv3((720, 1280, 3), 17, 3, num_scales=4, extra_ds=1)
model = net.compile_model(
    loss=yolo_loss(training_data.anchors, out_grids),
    optimizer='adam'
)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(Path.expanduser(Path("~/models/ror2_yolov3_v1"))),
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_best_only=True,)

try:
    model.fit(
        training_data,
        epochs=100,
        verbose=1,
        callbacks=[cp_callback]
    )
except KeyboardInterrupt:
    model.save(str(Path.expanduser(Path("~/models/ror2_yolov3_v1"))))
else:
    model.save(str(Path.expanduser(Path("~/models/ror2_yolov3_v1"))))