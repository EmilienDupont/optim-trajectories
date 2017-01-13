import numpy as np
import util

from keras import backend as K
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Flatten
from keras.models import Model

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

class TrajectoryVisualizer():
    def __init__(self):
        """
        """
        self.model = None
        self.callbacks = [util.LossHistory(), util.WeightHistory()]
        self.history = {}
        self.num_runs = 0
        self.projector = None
        self.start_weights = None

    def _reset_model(self, opt='sgd'):
        """
        Sets up model.

        Might be better to use a different model here...
        """
        input_img = Input(shape=(28, 28, 1))
        x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)
        x = Flatten()(x)
        output = Dense(10, activation='softmax')(x)
        self.model = Model(input=input_img, output=output)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, x_train, y_train, batch_size=128, num_epochs=5, repeat=1,
            reset=True, same_start=False, optimizers=['sgd']):
        """
        Parameters
        ----------
        x_train: numpy-array

        y_train: numpy-array

        batch_size: int

        num_epochs: int

        repeat: int
            Number of times you wish to repeat training.

        reset: bool
            If true resets the weights randomly at every repetition

        same_start: bool
            If true, always starts training with the same weights

        optimizers: list
            List of optimizers to use. E.g. [SGD(), Adam(), RMSprop(),
            SGD(momentum=0.9)]
        """
        for i, opt in enumerate(optimizers):

            print("\nOptimizer {}".format(i+1))

            for j in range(repeat):

                print("\nTraining {}".format(j+1))

                if reset or j == 0:
                    print("\tResetting model")
                    self._reset_model(opt=opt)

                if same_start:
                    print("\tWith same start")
                    if self.start_weights is None:
                        self.start_weights = self.model.get_weights()
                    else:
                        self.model.set_weights(self.start_weights)

                self.model.fit(x_train, y_train,
                               batch_size=batch_size,
                               nb_epoch=num_epochs,
                               callbacks=self.callbacks)
                self.history[self.num_runs] = {
                    'loss': self.callbacks[0].losses,
                    'weight': self.callbacks[1].w_history
                }
                self.num_runs += 1

    def set_projector(self):
        """
        Sets up Gaussian random matrix which will be used to project
        trajectories onto the plane.
        """
        stacked_histories = []
        for i in range(self.num_runs):
            stacked_histories.append(np.vstack(self.history[i]['weight']))

        self.projector = GaussianRandomProjection(n_components=2)
        self.projector.fit(np.vstack(stacked_histories))

    def project_weights(self, weights):
        """
        Parameters
        ----------
        weights: numpy-array
            weights you wish to project
        """
        return self.projector.transform(weights)

    def plot(self, subsample=1):
        """
        Plots all the trajectories currently saved in history.

        Parameters
        ----------
        subsample: int
            Subsample data so you only plot some of data. If subsample is e.g.
            4, only include every 4th point
        """
        if self.projector is None:
            self.set_projector()

        x_list = []
        y_list = []
        color_list = []
        for num_run in range(self.num_runs):
            weights = np.vstack(self.history[num_run]['weight'])
            losses = np.vstack(self.history[num_run]['loss']).squeeze()
            projected_weights = self.project_weights(weights)
            x_list.append(projected_weights[::subsample,0])
            y_list.append(projected_weights[::subsample,1])
            color_list.append(losses[::subsample])

        util.plot_2dscatter(x_list, y_list, color_list)
