from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from models.KerasModel import KerasModel


class TlResnet50(KerasModel):
    def __init__(self, input_shape):
        super(TlResnet50, self).__init__(input_shape)

    def build(self, input_shape):
        tl_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        last_layer = tl_model.output
        X = GlobalAveragePooling2D()(last_layer)
        X = Dense(512, activation='relu')(X)
        X = Dropout(0.2)(X)
        X = Dense(256, activation='relu')(X)
        X = Dropout(0.2)(X)
        X = Dense(1, activation='sigmoid')(X)

        model = Model(inputs=tl_model.input, outputs=X)
        for layer in model.layers[:-6]:
            layer.trainable = False
        return model
