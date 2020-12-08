# With help from: https://www.kaggle.com/kairess/lego-block-classification

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.layers import Input, Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, DepthwiseConv2D, LeakyReLU, Add, GlobalMaxPooling2D
from keras.models import Model
from skimage.transform import resize
import keras
import cv2
from PIL import Image, ImageOps

import glob, os, random

# Use GPU
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

base_path = '../../../lego_dataset/Cropped Images\\'

img_list = glob.glob(os.path.join(base_path, '*\\*.*'))

print(f"Number training images: {len(img_list)}")

# first = plt.imread(img_list[0])
# print(img_list[0])
# print(first.dtype)
# print(first.shape)
# plt.imshow(first)
# plt.show()


def resize_pad(img):
    old_size = img.shape[:2]
    ratio = 200. / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = resize(img, output_shape=new_size, mode='edge', preserve_range=True)

    delta_w = 200 - new_size[1]
    delta_h = 200 - new_size[0]
    padding = ((delta_h // 2, delta_h - (delta_h // 2)), (delta_w // 2, delta_w - (delta_w // 2)), (0, 0))

    img = np.pad(img, padding, 'edge')
    # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def preprocessing(x):
    x = resize_pad(x)
    return x


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing,
    rescale=1. / 255,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90,
    validation_split=0.1
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing,
    rescale=1. / 255,
    validation_split=0.1
)

train_generator = train_datagen.flow_from_directory(
    base_path,
    color_mode='rgb',
    target_size=(200, 200),
    subset='training',
    seed=0
)

validation_generator = test_datagen.flow_from_directory(
    base_path,
    color_mode='rgb',
    target_size=(200, 200),
    subset='validation',
    seed=0
)

# get labels
labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
print(labels)


# Import a Xception Model
base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)

# Create a Average Pooling after the CNN layers
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)

# Create a top Dense Layer
output = keras.layers.Dense(len(list(labels.keys())), activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)

# Train all layers in the model including pre-trained
for layer in base_model.layers:
    layer.trainable = True

# We set the optimizer to a Nesterov, good convergence quality, I haven't tested Nadam or RMSProp,
# the training is slow with the input size
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True, decay=0.001)

# We use accuracy and crossentropy to have distinct classes
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])

# train the model
# Steps = number of batches per epoch
history = model.fit_generator(
    train_generator,
    steps_per_epoch=int(32000/8),
    validation_data=validation_generator,
    validation_steps=int(3200/8),
    epochs=5
)
model.save('./model')
