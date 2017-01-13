import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
import time

from keras.callbacks import Callback
from keras.datasets import mnist
from keras.utils import np_utils
from plotly import tools
from random import shuffle


def load_process_mnist():
    """
    Hit me up with that normalized mnist stuff
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

def plot_2dscatter(x_list, y_list, color_list, filename='temp'):
    """
    """
    data = []

    for i, x in enumerate(x_list):
        y = y_list[i]
        color = color_list[i]
        trace = go.Scatter(
            x = x,
            y = y,
            mode = 'markers',
            marker=dict(
                color = color,
                colorscale = 'Viridis'
            )
        )
        data.append(trace)

    fig = go.Figure(data=data)
    py.plot(fig, auto_open=False,
            filename=get_timestamp_filename(filename))

def get_timestamp_filename(filename):
    """
    Returns filename with a timestap
    """
    date = time.strftime("%H-%M_%d-%m-%Y")
    return filename + '_' + date + '.html'

class LossHistory(Callback):
    """
    Callback to record loss after every batch.
    """
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class WeightHistory(Callback):
    """
    Callback to record weights in last layer after every batch.
    This could be changed?
    """
    def on_train_begin(self, logs={}):
        self.w_history = []
        self.b_history = []

    def on_batch_end(self, batch, logs={}):
        w, b = self.model.layers[-1].get_weights()
        self.w_history.append(w.reshape(-1))
        self.b_history.append(b)
