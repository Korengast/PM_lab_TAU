import Prepare_data
import numpy as np
from sklearn.model_selection import train_test_split
from models.FCModel import FCModel
from models.GaoetAl2017Model import GaoetAl2017Model
from models.BayramogluetAl2015Model import BayramogluetAl2015Model
from models.transfer_learning_resnet50 import TlResnet50
from models.transfer_learning_vgg16 import TlVGG16

EPOCHS = 2
BATCH_SIZE = 5

__author__ = "Koren Gast"

#data = Prepare_data.images_to_df('raw_data/')

data_dir = 'array_data/4D/'
print('loading data...')
X_train = np.load(data_dir+'X_train.npy')
print('...')
X_valid = np.load(data_dir+'X_valid.npy')
print('...')
y_train = np.load(data_dir+'y_train.npy')
print('...')
y_valid = np.load(data_dir+'y_valid.npy')
print('data loading completed')

# model = FCModel(X_train[0].shape, layers_num=1)
model = BayramogluetAl2015Model(X_train[0].shape)
# model = GaoetAl2017Model(X_train[0].shape)
# model = TlResnet50(X_train[0].shape)
# model = TlVGG16(X_train[0].shape)

model.compile()
model.model.summary()
model.fit(X_train, y_train, EPOCHS, BATCH_SIZE)
print('Model fit completed')
acc_train = model.evaluate(X_train, y_train)[1]
print()
print("Train Accuracy = " + str(acc_train))
acc_valid = model.evaluate(X_valid, y_valid)[1]
print("Valid Accuracy = " + str(acc_valid))

