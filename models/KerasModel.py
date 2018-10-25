from keras import optimizers
import numpy as np


class KerasModel(object):

    def __init__(self, input_shape, layers_num=0):
        self.layers_num = layers_num
        self.model = self.build(input_shape)

    def build(self, input_shape):
        pass

    def compile(self):
        adam = optimizers.adam()
        self.model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])

    def fit(self, x, y, epochs=10, batch_size=15, validation_data = None):
        if validation_data is None:
            return self.model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size, verbose=0)
        else:
            return self.model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size, validation_data=validation_data,
                                  verbose=0)

    def evaluate(self, x, y):
        loss_acc = self.model.evaluate(x=x, y=y)
        return loss_acc

    def predict(self, x):
        pred_prob = self.model.predict(x=x)
        preds = np.round(pred_prob)
        return preds

    def save_model_weights(self, name):
        self.model.save_weights('Weights/' + name)

    def load_model_weights(self, name):
        self.model.load_weights('Weights/' + name, by_name=False)
