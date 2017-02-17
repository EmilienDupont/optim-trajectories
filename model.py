from keras import backend as K
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Flatten
from keras.models import Model


class DeepModel():
    """Wrapper for Keras model."""
    def __init__(self):
        self.layer_shapes = None
        self.model = None
        self.start_weights = None
        self.opt = 'sgd'
        self._set_model()

    def _set_model(self):
        """
        """
        # Model architecture
        input_img = Input(shape=(28, 28, 1), name='input')
        x = Flatten()(input_img)
        x = Dense(512, activation='relu', name='weight_0')(x)
        output = Dense(10, activation='softmax', name='weight_1')(x)

        self.model = Model(input=input_img, output=output)
        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy',
                           metrics=['accuracy'])

        if self.start_weights is None:
            self.start_weights = self.model.get_weights()

        self._set_layer_shapes_dict()

    def reset_weights(self):
        """
        Sets weights in model to their initial values when the class was
        instantiated.
        """
        self.model.set_weights(self.start_weights)

    def get_layer_weight(self, layer_name):
        """
        Returns weight of a given layer.
        """
        layer = self.model.get_layer(layer_name)
        return layer.get_weights()[0]  # Only return weight, ignore bias

    def set_layer_weight(self, layer_name, weight):
        """
        Sets weight of a given layer.
        """
        layer = self.model.get_layer(layer_name)
        bias = layer.get_weights()[1]
        layer.set_weights([weight, bias])

    def _set_layer_shapes_dict(self):
        self.layer_shapes = {}
        for layer in self.model.layers:
            if layer.name.startswith('weight'):
                weight = layer.get_weights()[0]
                self.layer_shapes[layer.name] = weight.shape

    def reset_model(self, opt=None, reuse_weights=False):
        if opt is not None:
            self.opt = opt

        self._set_model()

        if reuse_weights:
            self.reset_weights()

    def fit(self, x_train, y_train, batch_size=128, num_epochs=5,
            callbacks=[]):
        if self.model is None:
            self._set_model()
        self.model.fit(x_train, y_train, batch_size=batch_size,
                       nb_epoch=num_epochs, callbacks=callbacks)

    def evaluate(self, x_test, y_test, batch_size=128):
        return self.model.evaluate(x_test, y_test, batch_size=batch_size,
                                   verbose=0)[0]
