import util
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from optim_trajectories import TrajectoryVisualizer
from model import DeepModel
from recorder import Recorder
from visualizer import Visualizer

x_train, y_train, x_test, y_test = util.load_process_mnist()
# Create model
model = DeepModel()
# Record different training paths of model
rec = Recorder(model, callbacks=[util.WeightHistory(-2)])
optimizers=[SGD(), SGD(momentum=0.9), RMSprop(), Adam()]
rec.fit(x_train, y_train, random_init=False, repeat=1, optimizers=optimizers)
# Visualize what has been recorded
viz = Visualizer(rec)
viz.plot_layer('weight_-1')