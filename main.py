import Prepare_data
import numpy as np
from models.FCModel import FCModel
from models.GaoetAl2017Model import GaoetAl2017Model
from models.BayramogluetAl2015Model import BayramogluetAl2015Model
from models.transfer_learning_resnet50 import TlResnet50
from models.transfer_learning_vgg16 import TlVGG16

EPOCHS = 10
BATCH_SIZE = 30

__author__ = "Koren Gast"

data = Prepare_data.images_to_df('raw_data/')
