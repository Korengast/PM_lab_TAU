from models.KerasModel import KerasModel
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
import numpy as np


class FCModel(KerasModel):
    def __init__(self, input_shape, layers_num):
        super(FCModel, self).__init__(input_shape, layers_num)

    def build(self, input_shape):
        X_input = Input(input_shape)
        X = Flatten()(X_input)
        for l in range(self.layers_num - 1, 0, -1):  # Hidden layers
            out = np.power(2, 2 * (l + 1))
            X = Dense(out, activation='sigmoid')(X)
            X = Dropout(0.2)(X)
        X = Dense(1, activation='sigmoid')(X)
        model = Model(inputs=X_input, outputs=X, name='FcModel_' + str(self.layers_num))
        return model
