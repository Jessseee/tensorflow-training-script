# Tensorflow model training notebook
The notebook included in the repository can be used as a starting point for a simple and clean way to train TensorFlow models for computer vision. **Tested on TensorFlow v2.8.0.**

## Config
In most cases the notebook itself should not need to be edited, instead a configuration file can be used to change the dataset, model, callbacks and training parameters. Below you can find a reference for all the possible options in the configuration file. *Note: \*required*

### Dataset*
| Args ||
:--                       | :-- 
src*                      | Root source directory of your dataset.
classes*                  | Names of the classes in the dataset.
class_mode                | Mode of the dataset classes (binary, categorical, etc.). Defaults to `categorical`.
batch                     | Size of the batches of data for training. Defaults to `32`.
class_options             | Additional parameters for intializing model class. See docs of the model class for more information.
compile_options           | Additional parameters for compiling model. See [docs](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).

### Model*
| Args ||
:--                        | :-- 
module*                    | Python module to import the model class from. See [docs](https://www.tensorflow.org/api_docs/python/tf/keras/applications) for a list of available models.
class*                     | Model class to import from the specified module.
weights                    | Specify pre-trained weights (imagenet, etc.) or path to TensorFlow weights file. Defaults to `None`.
name                       | Overwrite the name of the resulting model. Defaults to the name of the specified model.
checkpoints                | Path to directory to save checkpoints to. Defaults to `./models`.

### Optimizer*
| Args ||
:--                        | :-- 
module*                    | Python module to import the optimizer class from. See [docs](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) for a list of available optimizers.
class*                     | Optimizer class to import from the specified module.
options*                   | The constructor parameters of the specified optimizer.

### Callbacks
The notebook enables two callbacks by default:
- `PlotLearning`: This callback automatically plots the training metrics after each epoch.
- `ModelCheckpoint`: This callback is a build-in from tensorflow and automatically saves your model at the best performing epochs. Check [the docs](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) for more information.

Other callbacks can be imported from `tf.keras.callbacks`. See [docs](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks) for a list of available callbacks. Just including the name of the class (capitalized) as a key and the parameters as the values in the configuration file. For example:
```YAML
callbacks:
    ReduceLROnPlateau:
        monitor: val_loss
        mode: min
        patience: 5
        factor: 0.5
        min_lr: 0.000001
        verbose: 1
```

### Training
| Args ||
:--                        | :-- 
epochs                     | Number of epochs to train the model for.
training_steps_per_epoch   | Number of steps to take per epoch of training. Defaults to `train_gen.n//train_gen.batch_size`.
validation_steps_per_epoch | Number of steps to take per epoch of validation. Defaults to `valid_gen.n//valid_gen.batch_size`.
options                    | Additional parameters for the `Model.fit()` method. See [docs](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).
