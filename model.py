from keras import backend as K
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Flatten
from keras.models import Model

class DeepModel():
    def __init__(self):
        self.model = None
        self.start_weights = None
        self.opt = 'sgd'

    def _set_model(self):
        """
        """
        # Model architecture
        input_img = Input(shape=(28, 28, 1))
        x = Flatten()(input_img)
        x = Dense(512, activation='relu')(x)
        output = Dense(10, activation='softmax')(x)

        self.model = Model(input=input_img, output=output)
        self.model.compile(optimizer=self.opt, loss='categorical_crossentropy',
                           metrics=['accuracy'])

        if self.start_weights is None:
            self.start_weights = self.model.get_weights()

    def reset_weights(self):
        """
        Sets weights in model to their initial values when the class was
        instantiated.
        """
        self.model.set_weights(self.start_weights)

    def reset_model(self, opt=None, reuse_weights=False):
        if opt is not None:
            self.opt = opt

        self._set_model()

        if reuse_weights:
            self.reset_weights()

    def fit(self, x_train, y_train, batch_size=128, num_epochs=5, callbacks=[]):
        if self.model is None:
            self._set_model()
        self.model.fit(x_train, y_train, batch_size=batch_size,
                       nb_epoch=num_epochs, callbacks=callbacks)