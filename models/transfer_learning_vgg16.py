from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model

from models.KerasModel import KerasModel


class TlVGG16(KerasModel):
    def __init__(self, input_shape):
        super(TlVGG16, self).__init__(input_shape)

    def build(self, input_shape):
        tl_model = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
        last_layer = tl_model.get_layer('block5_pool').output
        X = Flatten()(last_layer)
        X = Dense(128, activation='relu')(X)
        X = Dropout(0.2)(X)
        X = Dense(128, activation='relu')(X)
        X = Dropout(0.2)(X)
        X = Dense(1, activation='sigmoid', name='output')(X)

        model = Model(inputs=tl_model.input, outputs=X)
        for layer in model.layers[:-3]:
            layer.trainable = False
        return model
