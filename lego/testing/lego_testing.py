from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import glob, os, random
import cv2


model = load_model('../models/model-80')


base_path = '../../../lego_dataset_orig/Cropped Images/'

img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))


def resize_pad(img):
    old_size = img.shape[:2]
    ratio = 200. / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = resize(img, output_shape=new_size, mode='edge', preserve_range=True)

    delta_w = 200 - new_size[1]
    delta_h = 200 - new_size[0]
    padding = ((delta_h // 2, delta_h - (delta_h // 2)), (delta_w // 2, delta_w - (delta_w // 2)), (0, 0))

    img = np.pad(img, padding, 'edge')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    plt.show()

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    return img


def preprocessing_val(x):
    x = resize_pad(x)
    return x


test_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_val,
    rescale=1. / 255,
    validation_split=0.1
)

validation_generator = test_datagen.flow_from_directory(
    base_path,
    color_mode='grayscale',
    target_size=(200, 200),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=0
)

results = model.evaluate_generator(validation_generator)
print("test loss, test acc:", results)
