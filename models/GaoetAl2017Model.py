from models.KerasModel import KerasModel
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, \
    Flatten, Dense, Dropout
from keras.models import Model

# Gao, Z., Wang, L., Zhou, L., & Zhang, J. (2017). HEp-2 cell image classification with deep convolutional neural networks. IEEE journal of biomedical and health informatics, 21(2), 416-428.‏

class GaoetAl2017Model(KerasModel):
    def __init__(self, input_shape):
        super(GaoetAl2017Model, self).__init__(input_shape)

    def build(self, input_shape):
        X_input = Input(input_shape)
        X = Conv2D(6, (7, 7), strides=(1, 1), activation='relu')(X_input)  # Not tanh
        X = BatchNormalization(axis=3)(X)
        X = MaxPooling2D((2, 2))(X)
        X = Conv2D(16, (4, 4), strides=(1, 1), activation='relu')(X)  # Not tanh
        X = BatchNormalization(axis=3)(X)
        X = MaxPooling2D((3, 3))(X)
        X = Conv2D(32, (3, 3), strides=(1, 1), activation='relu')(X)  # Not tanh
        X = BatchNormalization(axis=3)(X)
        X = MaxPooling2D((3, 3))(X)
        X = Flatten()(X)
        X = Dense(1, activation='sigmoid')(X)

        model = Model(inputs=X_input, outputs=X, name='GaoetAl2017Model')
        return model


