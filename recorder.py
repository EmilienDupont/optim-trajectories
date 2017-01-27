import util

class Recorder():
    def __init__(self, model, callbacks=[]):
        self.callbacks = [util.LossHistory(), util.WeightHistory(-1)] + callbacks
        self.callback_names = []
        for callback in self.callbacks:
            self.callback_names.append(callback.get_name())
        self.history = {}
        self.model = model
        self.num_runs = 0

    def fit(self, x_train, y_train, batch_size=128, num_epochs=5, repeat=1,
            random_init=True, optimizers=['sgd']):
        """
        Parameters
        ----------
        x_train : numpy-array

        y_train : numpy-array

        batch_size : int

        num_epochs : int

        repeat : int
            Number of times you wish to repeat training.

        random_init : bool
            If true initializes weights randomly at every repetition, if
            false starts with same weights at every repetition.

        optimizers : list
            List of optimizers to use. E.g. [SGD(), Adam(), RMSprop(),
            SGD(momentum=0.9)]
        """
        for i, opt in enumerate(optimizers):
            print("\nOptimizer {}/{}".format(i+1, len(optimizers)))

            for j in range(repeat):
                print("Rep {}/{}".format(j+1, repeat))

                self.model.reset_model(opt=opt, reuse_weights=(not random_init))

                self.model.fit(x_train, y_train,
                               batch_size=batch_size,
                               num_epochs=num_epochs,
                               callbacks=self.callbacks)

                # Record all data stored in callbacks
                one_run_history = {}
                for callback in self.callbacks:
                    one_run_history[callback.get_name()] = callback.get_data()
                self.history[self.num_runs] = one_run_history

                self.num_runs += 1