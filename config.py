from dataclasses import dataclass, field
from datetime import datetime
import yaml
import re


@dataclass
class Dataset:
    src: str
    classes: list
    class_mode: str = field(default="categorical")
    batch: int = field(default=32)
    train_options: dict = field(default_factory=dict)
    valid_options: dict = field(default_factory=dict)


@dataclass
class Model:
    cls: str = field(default="ResNet50")
    module: str = field(default="tensorflow.keras.applications")
    name: str = field(default=None)
    include_top: bool = field(default=False)
    weights: str = field(default="imagenet")
    input_shape: str = field(default="(244, 224, 3)")
    loss: str = field(default="binary_crossentropy")
    metrics: str | list | dict = field(default="accuracy")
    fc_layer_activation: str = field(default="sigmoid")
    class_options: dict = field(default_factory=dict)
    compile_options: dict = field(default_factory=dict)

    @property
    def input_tuple(self) -> tuple:
        return tuple(map(int, re.findall(r"(\d+)", self.input_shape)))


@dataclass
class Optimizer:
    cls: str
    module: str = field(default="tensorflow.keras.optimizers")
    options: dict = field(default_factory=dict)


@dataclass
class Schedule:
    cls: str
    module: str = field(default="tensorflow.keras.optimizers.schedules")
    options: dict = field(default_factory=dict)


@dataclass
class Callback:
    cls: str
    module: str = field(default="tensorflow.keras.callbacks")
    options: dict = field(default_factory=dict)


@dataclass
class Training:
    epochs: int = field(default=10)
    training_steps_per_epoch: int = field(default=None)
    validation_steps_per_epoch: int = field(default=None)
    options: dict = field(default_factory=dict)


class Config:
    def __init__(self, dataset, model, optimizer, schedule=None, callbacks=None, training=None):
        self.dataset = Dataset(**dataset)
        self.model = Model(**model)
        self.optimizer = Optimizer(**optimizer)
        self.schedule = Schedule(**(schedule or dict()))
        self.callbacks = [Callback(**c) for c in (callbacks or list())]
        self.training = Training(**(training or dict()))


def parse_config(path: str) -> Config:
    with open(path) as config_file:
        raw = config_file.read()
        unparsed = yaml.safe_load(raw)

    regex = {
        r'(\$now)': datetime.now().strftime("%Y%m%d-%H%M%S"),
        r'(\$model_class)': unparsed['model']['cls'].lower()
    }

    parsed = raw
    for pattern, repl in regex.items():
        parsed = re.sub(pattern, repl, parsed)

    return Config(**yaml.safe_load(parsed))
