from os import listdir
from PIL import Image
from random import random
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('ggplot')

############# Explore the data ##############

MTs = []
WTs = []
MT_modes = []
WT_modes = []
MT_sizes = []
WT_sizes = []
MT_avg_R = []
MT_avg_G = []
MT_avg_B = []
MT_avg_A = []
WT_avg_R = []
WT_avg_G = []
WT_avg_B = []
WT_avg_A = []
for MT_p, WT_p in zip(listdir('DL_task/MT/'), listdir('DL_task/WT/')):
    MT_img = Image.open('DL_task/MT/' + MT_p)
    WT_img = Image.open('DL_task/WT/' + WT_p)
    MTs.append(MT_img)
    WTs.append(WT_img)
    MT_modes.append(MT_img.mode)
    WT_modes.append(WT_img.mode)
    MT_sizes.append(MT_img.size)
    WT_sizes.append(WT_img.size)
    MT_rgba = np.array(MT_img).mean(axis=(0, 1))
    WT_rgba = np.array(WT_img).mean(axis=(0, 1))
    for i in range(4):
        MT_avg_R.append(MT_rgba[0])
        WT_avg_R.append(WT_rgba[0])
        MT_avg_G.append(MT_rgba[1])
        WT_avg_G.append(WT_rgba[1])
        MT_avg_B.append(MT_rgba[2])
        WT_avg_B.append(WT_rgba[2])
        try:
            MT_avg_A.append(MT_rgba[3])
        except:
            MT_avg_A.append(-1)
        try:
            WT_avg_A.append(WT_rgba[3])
        except:
            WT_avg_A.append(-1)

## Modes
plt.subplot(2, 1, 1)
plt.bar(x=range(2), height=[MT_modes.count('RGB'), MT_modes.count('RGBA')], tick_label=('RGB', 'RGBA'))
plt.title('Modes')
plt.ylabel('MT')

plt.subplot(2, 1, 2)
plt.bar(x=range(2), height=[WT_modes.count('RGB'), WT_modes.count('RGBA')], tick_label=('RGB', 'RGBA'))
plt.xlabel('mode')
plt.ylabel('WT')

plt.show()

## Size
plt.subplot(2, 1, 1)
plt.hist(MT_sizes[0], bins=40, range=(0, 4000))
plt.title('Picture size')
plt.ylabel('MT')

plt.subplot(2, 1, 2)
plt.hist(WT_sizes[0], bins=40, range=(0, 4000))
plt.xlabel('size')
plt.ylabel('MT')

plt.show()

## Red
plt.subplot(2, 1, 1)
plt.hist(MT_avg_R, bins=255, range=(0, 255))
plt.title('RED')
plt.ylabel('MT')

plt.subplot(2, 1, 2)
plt.hist(WT_avg_R, bins=255, range=(0, 255))
plt.xlabel('avg')
plt.ylabel('MT')

plt.show()

## Red zoom
plt.subplot(2, 1, 1)
plt.hist(MT_avg_R, bins=60, range=(0, 6))
plt.title('RED_zoom_in')
plt.ylabel('MT')

plt.subplot(2, 1, 2)
plt.hist(WT_avg_R, bins=60, range=(0, 6))
plt.xlabel('avg')
plt.ylabel('MT')

plt.show()

## Green
plt.subplot(2, 1, 1)
plt.hist(MT_avg_G, bins=255, range=(0, 255))
plt.title('GREEN')
plt.ylabel('MT')

plt.subplot(2, 1, 2)
plt.hist(WT_avg_G, bins=255, range=(0, 255))
plt.xlabel('avg')
plt.ylabel('MT')

plt.show()

## Green_zoom
plt.subplot(2, 1, 1)
plt.hist(MT_avg_G, bins=60, range=(0, 6))
plt.title('GREEN_zoom')
plt.ylabel('MT')

plt.subplot(2, 1, 2)
plt.hist(WT_avg_G, bins=60, range=(0, 6))
plt.xlabel('avg')
plt.ylabel('MT')

plt.show()

## Blue
plt.subplot(2, 1, 1)
plt.hist(MT_avg_B, bins=255, range=(0, 255))
plt.title('BLUE')
plt.ylabel('MT')

plt.subplot(2, 1, 2)
plt.hist(WT_avg_B, bins=255, range=(0, 255))
plt.xlabel('avg')
plt.ylabel('MT')

plt.show()

## Blue zoom
plt.subplot(2, 1, 1)
plt.hist(MT_avg_B, bins=60, range=(0, 6))
plt.title('BLUE_zoom')
plt.ylabel('MT')

plt.subplot(2, 1, 2)
plt.hist(WT_avg_B, bins=60, range=(0, 6))
plt.xlabel('avg')
plt.ylabel('MT')

plt.show()

## Alpha
plt.subplot(2, 1, 1)
plt.hist(MT_avg_A, bins=258, range=(-2, 256))
plt.title('ALPHA')
plt.ylabel('MT')

plt.subplot(2, 1, 2)
plt.hist(WT_avg_A, bins=258, range=(-2, 256))
plt.xlabel('avg')
plt.ylabel('MT')

plt.show()

############# Prepare the data ##############

resolutions = [int(3840 / 2), int(3840 / 10), int(3840 / 20)]
augmentations = ['no', 'flip', 'rotate']

X_train = {'1920_no': [], '384_flip': [], '192_rotate': []}
Y_train = {'1920_no': [], '384_flip': [], '192_rotate': []}
X_valid = {'1920_no': [], '384_flip': [], '192_rotate': []}
Y_valid = {'1920_no': [], '384_flip': [], '192_rotate': []}

res_aug_maps = {'1920_no': [resolutions[0], augmentations[0]],
                '384_flip': [resolutions[1], augmentations[1]],
                '192_rotate': [resolutions[2], augmentations[2]]}


def appends(M, W, x, y):
    x.append(M)
    y.append(1)
    x.append(W)
    y.append(0)


def augmenting(img, type):
    imgs = []
    if type == 'no':
        imgs.append(img)
    if type == 'flip':
        imgs.append(img)
        imgs.append(np.flip(img, 1))
    if type == 'rotate':
        imgs.append(img)
        imgs.append(np.rot90(img))
        imgs.append(np.rot90(np.rot90(img)))
        imgs.append(np.rot90(np.rot90(np.rot90(img))))
    return imgs


for res_aug in res_aug_maps.keys():
    for MT_p, WT_p in zip(listdir('DL_task/MT/'), listdir('DL_task/WT/')):
        MT_img = Image.open('DL_task/MT/' + MT_p)
        WT_img = Image.open('DL_task/WT/' + WT_p)
        MT_img = MT_img.convert('RGB')
        MT_img = MT_img.resize((res_aug_maps[res_aug][0], res_aug_maps[res_aug][0]))
        WT_img = WT_img.convert('RGB')
        WT_img = WT_img.resize((res_aug_maps[res_aug][0], res_aug_maps[res_aug][0]))

        if random() > 0.2:
            MT_augments = augmenting(np.array(MT_img), res_aug_maps[res_aug][1])
            WT_augments = augmenting(np.array(WT_img), res_aug_maps[res_aug][1])
            for MT_a, WT_a in zip(MT_augments, WT_augments):
                appends(MT_a, WT_a, X_train[res_aug], Y_train[res_aug])
        else:
            MT_augments = augmenting(np.array(MT_img), res_aug_maps[res_aug][1])
            WT_augments = augmenting(np.array(WT_img), res_aug_maps[res_aug][1])
            for MT_a, WT_a in zip(MT_augments, WT_augments):
                appends(MT_a, WT_a, X_valid[res_aug], Y_valid[res_aug])

for res_aug in res_aug_maps.keys():
    xtrain = np.array(X_train[res_aug])
    xvalid = np.array(X_valid[res_aug])
    ytrain = np.array(Y_train[res_aug])
    yvalid = np.array(Y_valid[res_aug])
    xtrain = xtrain / 255.
    xvalid = xvalid / 255.
    np.save(res_aug + '_X_train', xtrain)
    np.save(res_aug + '_X_valid', xvalid)
    np.save(res_aug + '_Y_train', ytrain)
    np.save(res_aug + '_Y_valid', yvalid)

X_test = {'1920': [], '384': [], '192': []}
test_file_names = {'1920': [], '384': [], '192': []}
for res in resolutions:
    for p in listdir('DL_task/test/'):
        img = Image.open('DL_task/test/' + p)
        img = img.convert('RGB')
        img = img.resize((res, res))
        X_test[str(res)].append(np.array(img))
        test_file_names[str(res)].append(p)

for res in resolutions:
    xtest = np.array(X_test[str(res)])
    xtest = xtest / 255.
    np.save(str(res) + '_X_test', xtest)
    np.save(str(res) + 'test_file_names', test_file_names[str(res)])
