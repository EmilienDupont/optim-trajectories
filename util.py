import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
import time

from keras.callbacks import Callback
from keras.datasets import mnist
from keras.utils import np_utils
from plotly import tools
from random import shuffle

# list of categorical colors
cat_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


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
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                color=color,
                colorscale='Viridis'
            )
        )
        data.append(trace)

    fig = go.Figure(data=data)
    py.plot(fig, filename=get_timestamp_filename(filename))


def plot(x, y, title='', x_axis='x', y_name=None, filename='temp_plot',
         colors=None):
    """
    Simple x, y plot.

    Parameters
    ----------
    x : array
        Array defining x-axis

    y : array or list of arrays
        If y is a single array, plot traditional x, y plot. If is a list of
        arrays, overlay all these on the same plot.

    y_name : string or list
        If y is a single array, string name for y, otherwise list of strings
        for each y.

    colors : list or array
        If plotting several arrays and some arrays belong to a particular
        category, put different colors on the plots. Colors is an array of
        integers in the range 0 to number of categories - 1
    """
    if type(y) is list or type(y) is tuple:
        # If each y hasn't been named, just call it 1, 2, 3...
        if y_name is None:
            y_name = range(len(y))

        traces = []
        for i, y_sub in enumerate(y):
            if colors is None:
                marker = {'color': cat_colors[i % 10]}
            else:
                marker = {'color': cat_colors[colors[i]]}
            traces.append(
                go.Scatter(
                    x=x,
                    y=y_sub,
                    mode='lines',
                    name=y_name[i],
                    line=marker
                )
            )

    else:
        if y_name is None:
            y_name = 'y'
        trace = go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=y_name
        )
        traces = [trace]

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=x_axis
        )
    )

    fig = go.Figure(data=traces, layout=layout)
    py.plot(fig, filename=get_timestamp_filename(filename))


def get_timestamp_filename(filename):
    """
    Returns filename with a timestap
    """
    date = time.strftime("%H-%M_%d-%m-%Y")
    return filename + '_' + date + '.html'


def get_xy_and_grid(grid_size=10, x_range=(-2, 2), y_range=(-2, 2)):
    """
    Returns x, y and grid.
    """
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(x_range[0], x_range[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    return x, y, xx, yy


def heatmap(x, y, z, filename='temp'):
    """
    Heatmap plots
    """
    trace = [go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale='Viridis',
                zsmooth='best'
            )]
    fig = go.Figure(data=trace)
    py.plot(fig, filename=get_timestamp_filename(filename))


class LossHistory(Callback):
    """
    Callback to record loss after every batch.
    """
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def get_data(self):
        return self.losses

    def get_name(self):
        return "loss"


class WeightHistory(Callback):
    """
    Callback to record weights in a particular layer after every batch.
    """
    def __init__(self, layer):
        self.layer = layer

    def on_train_begin(self, logs={}):
        self.w_history = []
        self.b_history = []

    def on_batch_end(self, batch, logs={}):
        layer = self.model.get_layer(self.layer)
        w, b = layer.get_weights()
        self.w_history.append(w.flatten())
        self.b_history.append(b)

    def get_data(self):
        return self.w_history

    def get_name(self):
        return self.layer
