from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from skimage.transform import resize
import glob, os, random


model = load_model('./model')


base_path = '..\\lego_dataset\\cropped images\\'

img_list = glob.glob(os.path.join(base_path, '*\\*.jpg'))


def resize_pad(img):
    old_size = img.shape[:2]
    ratio = 200. / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = resize(img, output_shape=new_size, mode='edge', preserve_range=True)

    delta_w = 200 - new_size[1]
    delta_h = 200 - new_size[0]
    padding = ((delta_h // 2, delta_h - (delta_h // 2)), (delta_w // 2, delta_w - (delta_w // 2)), (0, 0))

    img = np.pad(img, padding, 'edge')

    return img


def preprocessing_train(x):
    x = resize_pad(x)
    return x


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
