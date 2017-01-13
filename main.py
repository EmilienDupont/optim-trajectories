from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from optim_trajectories import TrajectoryVisualizer
from util import load_process_mnist

x_train, y_train, x_test, y_test = load_process_mnist()

optimizers=[SGD()]#, SGD(momentum=0.9), RMSprop(), Adam()]

trajectories = TrajectoryVisualizer()
# Record loss and weight histories for 2 instances of each optimizer
trajectories.fit(x_train, y_train, repeat=2, optimizers=optimizers)
# Project and plot the trajectories obtained
trajectories.plot()
