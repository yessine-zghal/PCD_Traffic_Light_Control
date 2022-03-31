import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

# Keras GPU utilization settings
import tensorflow as tf

# Avoid warning about tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.compat.v1.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# To log device placement (on which device the operation ran)
# config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)


class TrainModel2:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._width = width
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)
        self._training_loss = 0


    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        return model

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)

    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states)

    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        history = self._model.fit(states, q_sa, epochs=1, verbose=0)
        # Get training and testing loss to analyse the model behavior
        self._training_loss = history.history['loss'][0]

    def save_model(self, path, num):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model_' + str(num) + '.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True,
                   show_layer_names=True)

    @property
    def training_loss(self):
        return self._training_loss

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path, num):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path, num)

    def _load_my_model(self, model_folder_path, num):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model_' + str(num) + '.h5')

        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)

    @property
    def input_dim(self):
        return self._input_dim