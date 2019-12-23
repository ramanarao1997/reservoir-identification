import os
import shutil
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

directories = ['train', 'validate']
categories = ['reservoirs', 'terrains']
augment_n = [30, 30]

datagen = ImageDataGenerator(rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,
    shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')

# Replace with your project's root folder path
root = 'C:/Users/ramana/Desktop/Reservoir Identification/data/'

for dir in directories:
    for j in range(len(categories)):
        cat = categories[j]
        n = augment_n[j]
        path = root + dir + '/' + cat
        image_list = os.listdir(path)
        for file in image_list:
            img = load_img(path = (path + '/' + file),  target_size = (224, 224))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in datagen.flow(x, batch_size = 1, save_to_dir = (root + dir + '/' + cat), save_prefix = 'aug_', save_format = 'TIF'):
                i += 1
                if i >= n:
                    break