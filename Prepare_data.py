from os import listdir
from PIL import Image
import pandas as pd
import numpy as np

__author__ = "Koren Gast"


## Save all images in dataframe including meta-data
def images_to_df(directory):
    columns_list = ['image_array', 'well', 'field', 'dye_type']
    raw_data_df = pd.DataFrame(columns=columns_list)
    for img_filename in listdir(directory):
        if img_filename.endswith('tif'):
            img = Image.open(directory + img_filename)
            img = np.array(img)
            well = img_filename[0:6]  # wells: B-3, C-3, B-9, C-9
            field = img_filename[11:13]  # 1 to 16
            last_space = img_filename.rfind(' ')
            dye_type = img_filename[last_space + 1:-5]  # dye_types: Cy3, Cy5, DAPI, FITC
            raw_data_df = raw_data_df.append({'image_array': img,
                                              'well': well,
                                              'field': field,
                                              'dye_type': dye_type},
                                             ignore_index=True)
    return raw_data_df


# img_df = images_to_df('raw_data/')

