from os import listdir
from PIL import Image
from random import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

__author__ = "Koren Gast"

plt.style.use('ggplot')

## Save all images in dataframe including meta-data

columns_list = ['tiff_image_file', 'well', 'field', 'dye_type']
raw_data_df = pd.DataFrame(columns=columns_list)

for img_filename in listdir('raw_data/'):
    if img_filename.endswith('tif'):
        img = Image.open('raw_data/' + img_filename)
        well = img_filename[0:6]    # wells: B-3, C-3, B-9, C-9
        field = img_filename[11:13] # 1 to 16
        last_space = img_filename.rfind(' ')
        dye_type = img_filename[last_space + 1:-5] # dye_types: Cy3, Cy5, DAPI, FITC
        raw_data_df = raw_data_df.append({'tiff_image_file': img,
                                          'well': well,
                                          'field': field,
                                          'dye_type': dye_type},
                                         ignore_index=True)



