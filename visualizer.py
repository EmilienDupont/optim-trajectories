import numpy as np
import util
from sklearn.random_projection import GaussianRandomProjection

class Visualizer():
    def __init__(self, recorder):
        self.projectors = None
        self.rec = recorder

    def set_weight_projectors(self):
        """
        Sets up Gaussian random matrix which will be used to project
        trajectories onto the plane.
        We require a projector for every weight matrix as they might
        have different shapes.
        """
        self.projectors = {}
        for callback_name in self.rec.callback_names:
            if not callback_name.startswith('weight'): continue
            projector = GaussianRandomProjection(n_components=2)
            # Fit method only requires the shape of the array, so it doesn't
            # matter from which run we pick weight matrix
            weight = self.rec.history[0][callback_name][0]
            projector.fit(weight.reshape(1, -1))
            self.projectors[callback_name] = projector

    def project_weights(self, callback_name, weights):
        """
        Parameters
        ----------
        callback_name : string
            identifier for callback. E.g. 'weight_2'

        weights : numpy-array
            weights you wish to project
        """
        return self.projectors[callback_name].transform(weights)

    def plot_layer(self, callback_name, subsample=2):
        """
        Plots all the trajectories of the layer specified by callback_name
        currently saved in history. Creates a color gradient with respect
        to loss.

        Parameters
        ----------
        callback_name : string
            identifier for callback. E.g. 'weight_2'

        subsample : int
            Subsample data so you only plot some of data. If subsample is e.g.
            4, only include every 4th point
        """
        if self.projectors is None:
            self.set_weight_projectors()

        x_list = []
        y_list = []
        color_list = []
        for num_run in range(self.rec.num_runs):
            weights = np.vstack(self.rec.history[num_run][callback_name])
            losses = np.vstack(self.rec.history[num_run]['loss']).squeeze()
            projected_weights = self.project_weights(callback_name, weights)
            x_list.append(projected_weights[::subsample,0])
            y_list.append(projected_weights[::subsample,1])
            color_list.append(losses[::subsample])

        util.plot_2dscatter(x_list, y_list, color_list, filename=callback_name)