from models.KerasModel import KerasModel
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, \
    Flatten, Dense, Dropout
from keras.models import Model

# Bayramoglu, N., Kannala, J., & Heikkilä, J. (2015, November). Human epithelial type 2 cell classification with convolutional neural networks. In Bioinformatics and Bioengineering (BIBE), 2015 IEEE 15th International Conference on (pp. 1-6). IEEE.‏

class BayramogluetAl2015Model(KerasModel):
    def __init__(self, input_shape):
        super(BayramogluetAl2015Model, self).__init__(input_shape)

    def build(self, input_shape):
        X_input = Input(input_shape)
        X = AveragePooling2D((20, 20))(X_input) # Reduce the image size
        X = Conv2D(32, (5, 5), strides=(1, 1))(X)
        X = MaxPooling2D((3, 3))(X)
        X = Conv2D(32, (5, 5), strides=(1, 1), activation='relu')(X)
        X = AveragePooling2D((3, 3))(X)
        X = Conv2D(64, (5, 5), strides=(1, 1), activation='relu')(X)
        X = AveragePooling2D((3, 3))(X)
        X = Flatten()(X)
        X = Dense(1, activation='sigmoid')(X)

        model = Model(inputs=X_input, outputs=X, name='BayramogluetAl2017Model')
        return model


