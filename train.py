from argparse import ArgumentParser
import os

from config import parse_config

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(f"\nTensorFlow v{tf.__version__}")
print(f"GPU Device: {tf.test.gpu_device_name() or 'NOT AVAILABLE'}")


# --------------- Config ---------------
parser = ArgumentParser(prog='train.py')
parser.add_argument('--config', '--cfg', '-c', type=str, default='configs/config.yaml')
args = parser.parse_args()

config = parse_config(args.config)


# --------------- Dataset ---------------
print(f"\nDatasets: {config.dataset.src}")
gen_train = ImageDataGenerator().flow_from_directory(
    os.path.join(config.dataset.src, 'training'),
    batch_size=config.dataset.batch,
    class_mode=config.dataset.class_mode,
    **config.dataset.train_options
)

gen_valid = ImageDataGenerator().flow_from_directory(
    os.path.join(config.dataset.src, 'validation'),
    batch_size=config.dataset.batch,
    class_mode=config.dataset.class_mode,
    **config.dataset.valid_options
)

print()


# --------------- Model ---------------

ModelClass = getattr(
    __import__(
        config.model.module,
        fromlist=[config.model.cls]
    ),  config.model.cls
)

base = ModelClass(
    include_top=config.model.include_top,
    weights=config.model.weights,
    input_shape=config.model.input_tuple,
    **config.model.class_options
)

model = Sequential()
model._name = config.model.name or base.name
model.add(base)
model.add(GlobalAveragePooling2D())
model.add(
    Dense(
        len(config.dataset.classes)-1,
        activation=config.model.fc_layer_activation
    )
)
model.layers[0].trainable = False

model.summary()
print()

# --------------- Optimizer ---------------
optimizer_options = config.optimizer.options

if config.schedule is not None:
    ScheduleClass = getattr(
        __import__(
            config.schedule.module,
            fromlist=[config.schedule.cls]
        ),  config.schedule.cls
    )

    lr_schedule = ScheduleClass(**config.schedule.options)
    optimizer_options.lr = lr_schedule

OptimizerClass = getattr(
    __import__(
        config.optimizer.module,
        fromlist=[config.optimizer.cls]
    ),  config.optimizer.cls
)

optimizer = OptimizerClass(**optimizer_options)

print('optimizer:', OptimizerClass.__name__)


# --------------- Compiling ---------------
model.compile(
    optimizer=optimizer,
    loss=config.model.loss,
    metrics=config.model.metrics,
    **config.model.compile_options
)


# --------------- Callbacks ---------------
callbacks = []

if config.callbacks is not None:
    for callback in config.callbacks:
        CallbackClass = getattr(
            __import__(
                callback.module,
                fromlist=[callback.cls]
            ),  callback.cls
        )
        callbacks.append(CallbackClass(**callback.options))

print(f"callbacks: {[callback.cls for callback in config.callbacks]}\n")


# --------------- training ---------------
model.fit(
    gen_train,
    steps_per_epoch=config.training.training_steps_per_epoch,
    validation_data=gen_valid,
    validation_steps=config.training.validation_steps_per_epoch,
    callbacks=callbacks,
    epochs=config.training.epochs,
    **config.training.options
)
