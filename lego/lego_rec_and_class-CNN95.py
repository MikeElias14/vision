import cv2
# need to use TF here as thats what the model used
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from classes_CNN95 import classes
import os, glob
import random


def show_images(imgs):
    fig = plt.figure()
    for i in range(len(imgs)):
        fig.add_subplot(2, 2, i+1)
        plt.imshow(imgs[i])
    plt.show()


def crop_image(img, img_bg):

    # Gray
    img_bg_gray = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Diff and blur
    diff_gray = cv2.absdiff(img_bg_gray, img_gray)
    diff_gray_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)

    # Sharpen
    sharpen_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_sharp = cv2.filter2D(diff_gray_blur, -1, sharpen_filter)

    # find otsu's threshold value with OpenCV function
    ret, img_thresh = cv2.threshold(img_sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # let's now draw the contour
    arr_cnt, a1 = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # let's copy the example image, so we don't paint over it
    img_with_allcontours = img.copy()
    cv2.drawContours(img_with_allcontours, arr_cnt, -1, (0, 255, 0), 3)

    # get the dimensions of the image
    height, width, channels = img.shape

    validcontours = []
    contour_index = -1

    # iterate through each contour found
    for i in arr_cnt:
        contour_index += 1
        ca = cv2.contourArea(i)

        # Calculate W/H Ratio of image
        x, y, w, h = cv2.boundingRect(i)
        aspect_ratio = float(w) / h

        # Contours at the edges of the image are most likely not valid contours
        edge_noise = False
        # if contour starts at x=0 then it's on th edge
        if x == 0:
            edge_noise = True
        if y == 0:
            edge_noise = True

        # if the contour x value + its contour width exceeds image width, it is on an edge
        if (x + w) == width:
            edge_noise = True
        if (y + h) == height:
            edge_noise = True

        # DISCARD noise with measure by area (1x1 round plate dimensions is 1300)
        # if by any chance a contour is drawn on one pixel, this catches it.
        # TODO: Tweak this value, better camera, higher thresh
        if ca > 2500:
            print(f"Contour Area: {ca}")

            # DISCARD if at the Edge
            if not edge_noise:
                validcontours.append(contour_index)

    # copy the original picture
    img_with_contours = img.copy()
    print(f"{len(validcontours)} objects detected")

    for i in validcontours:
        cv2.drawContours(img_with_contours, arr_cnt, i, (0, 255, 0), 3)


    # Generate mask
    img_inverse = cv2.bitwise_not(img_sharp)
    mask = np.ones(img_inverse.shape)
    for i in validcontours:
        mask = cv2.drawContours(mask, arr_cnt, i, 0, cv2.FILLED)

    # apply_mask
    img_masked = img_inverse.copy()
    img_masked[mask.astype(np.bool)] = 0

    # Crop the img
    imgs_rects = []
    for i in validcontours:
        x, y, w, h = cv2.boundingRect(arr_cnt[i])
        # Make cropped image larger than bonding rectangle
        # 2x4 is 700x700,
        delta_y = int((1000-h)/2)
        delta_x = int((1000-w)/2)
        cropped_image = cv2.cvtColor(img_masked[y-delta_y:y+h+delta_y, x-delta_x:x+w+delta_x], cv2.COLOR_GRAY2RGB)
        # cropped_image = img_masked[y-delta_y:y+h+delta_y, x-delta_x:x+w+delta_x]
        imgs_rects.append([cropped_image, [x, y, x+w, y+h]])
    return imgs_rects


def draw_class(img, class_rect: dict, target_img):
    for key in class_rect:
        color = (255, 0, 0)
        if key == target_img:
            color = (0, 255, 0)
        x, y, x2, y2 = class_rect.get(key)
        cv2.rectangle(img, (x, y), (x2, y2), color, 4)
        cv2.putText(img, key, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 4)
    return img


def main():
    # Load image
    base_path = '../../lego_dataset/my_images/'
    img_list = glob.glob(os.path.join(base_path, '*'))
    # img_path = img_list[random.randrange(len(img_list))]
    # img_path = '../../lego_dataset/my_images/PXL_20201209_175143762.jpg'
    # img_path = '../../lego_dataset/my_images/PXL_20201209_175130757.jpg'
    # img_path = '../../lego_dataset/my_images/PXL_20201209_184337353.jpg'
    img_path = '../../lego_dataset/my_images/assort_grey.jpg'

    # Load model
    model = load_model('./models/lego_CNN_95.h5')

    img_bg = cv2.imread('../../lego_dataset/my_images/background.jpg')
    img_orig = cv2.imread(img_path)
    imgs_rects = crop_image(img_orig, img_bg)
    class_rects = {}

    # define the target brick
    # target = '3020 - 2x4 Plate'
    # target = '3003 - 2x2 Brick'
    target = '3001 - 2x4 Brick'
    target_key = list(classes.keys())[list(classes.values()).index(target)]
    target_img = []
    target_confidence = 0

    imgs_to_predict = []
    rects = []

    # data of form [img, [x,y,x1,x2]]
    for img_rect in imgs_rects:
        img = tf.image.convert_image_dtype(img_rect[0], tf.float32)
        img_resized = tf.image.resize(img, [400, 400])
        img_expanded = np.expand_dims(img_resized, axis=0)
        imgs_to_predict.append(img_expanded)
        rects.append(img_rect[1])

    # Get prediction
    predictions = model.predict(np.array(imgs_to_predict))

    for score in predictions:
        classification = classes.get(np.argmax(score))
        confidence = round(np.max(score) * 100, 2)

        # Human readable prediction and assign rect
        class_key = f"{classification} - {confidence}%"
        class_rects[class_key] = rects.pop(0)
        print(class_key)

        # Set the target image key
        if score[target_key] > target_confidence:
            target_confidence = score[target_key]
            target_img = class_key

    img_classed = draw_class(img_orig, class_rects, target_img)

    plt.imshow(img_classed)
    plt.show()


if __name__ == '__main__':
    main()

