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
            img = np.array(img) / (2 ** 16)  # Pixels value ranges from 0 to 2^16
            well = img_filename[0:6]  # wells: B-3, C-3, B-9, C-9
            field = img_filename[11:13]  # 1 to 16
            last_space = img_filename.rfind(' ')
            dye_type = img_filename[last_space + 1:-5]  # dye_types: Cy3, Cy5, DAPI, FITC
            raw_data_df = raw_data_df.append({'image_array': img,
                                              'well/field': well + '/' + field,
                                              'dye_type': dye_type},
                                             ignore_index=True)
    raw_data_df = raw_data_df.groupby(['well/field'])['image_array'].apply(stack_arrays)
    raw_data_df = pd.DataFrame({'well/field': raw_data_df.index, 'image_array': raw_data_df.values})
    raw_data_df['is_ill'] = raw_data_df['well/field'].apply(lambda w: 1 if w.startswith('C - 03') or
                                                                           w.startswith('B - 03') else 0)
    return raw_data_df


def stack_arrays(ser):
    return np.stack(ser)

# img_df = images_to_df('raw_data/')
