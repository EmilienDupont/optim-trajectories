import numpy as np
import util
from sklearn.random_projection import GaussianRandomProjection


class Visualizer():
    """
    Visualize projections, optimization trajectories and many other aspects of
    model.
    """
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

    def _invert_point(self, layer_name, point):
        """
        Solve the underdetermined system y = P * x, where P is the projection
        matrix, y is the projection and x is the weight vector we solve for.
        """
        # Get random projection matrix
        projection_matrix = self.projectors[layer_name].components_
        # Get pseudoinverse of projection
        pseudo_inv_projection = np.linalg.pinv(projection_matrix)
        # Return weight corresponding to projected point
        return np.dot(pseudo_inv_projection, point)

    def _get_loss_at_point(self, x_train, y_train, point, layer_name):
        """
        """
        # Find the weight corresponding to projected point
        inv_weight = self._invert_point(layer_name, point)
        # Reshape weight to its original dimensions
        orig_shape = self.rec.model.layer_shapes[layer_name]
        inv_weight = inv_weight.reshape(orig_shape)
        # Update model with computed weight
        self.rec.model.set_layer_weight(layer_name, inv_weight)
        # Compute loss with updated model
        return self.rec.model.evaluate(x_train, y_train)

    def _get_loss_on_grid(self, x_train, y_train, layer_name, xx, yy):
        """
        Iterate over a grid of points defined by xx and yy and determines loss
        at that given point in weight space.
        """
        grid_size = xx.shape[0]
        loss = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                point = np.array((xx[i, j], yy[i, j]))
                loss[i, j] = self._get_loss_at_point(x_train, y_train, point,
                                                     layer_name)
                print((i, j, loss[i, j]))
        return loss

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
            x_list.append(projected_weights[::subsample, 0])
            y_list.append(projected_weights[::subsample, 1])
            color_list.append(losses[::subsample])

        util.plot_2dscatter(x_list, y_list, color_list, filename=callback_name)

    def plot_loss(self):
        """
        """
        losses = []
        for num_run in range(self.rec.num_runs):
            losses.append(np.array(self.rec.history[num_run]['loss']))
        num_steps = losses[0].shape[0]
        util.plot(np.arange(num_steps), losses, filename='loss')

    def plot_loss_heatmap(self, x_train, y_train, layer_name, grid_size=10):
        """
        Iterates over a 2d grid, and for each point uses the pseudoinverse of
        the projection matrix to find the minimum norm weight matrix
        corresponding to this point. Then computes the loss corresponding to
        that particular setting of the matrix, given that all the other layers
        are constant.
        """
        if self.projectors is None:
            self.set_weight_projectors()
        x, y, xx, yy = util.get_xy_and_grid(grid_size)
        loss = self._get_loss_on_grid(x_train, y_train, layer_name, xx, yy)
        util.heatmap(x, y, loss)