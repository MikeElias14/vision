import cv2
# use Tf backend here since its faster when loading since trials
from tensorflow.keras.models import load_model
from tensorflow import keras
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import random
from classes_model80 import classes


def prep_img(img):
    old_size = img.shape[:2]
    ratio = 200. / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    img = resize(img, output_shape=new_size, mode='edge', preserve_range=True)

    delta_w = 200 - new_size[1]
    delta_h = 200 - new_size[0]
    padding = ((delta_h // 2, delta_h - (delta_h // 2)), (delta_w // 2, delta_w - (delta_w // 2)), (0, 0))

    img = np.pad(img, padding, 'edge')
    img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=-1)
    img = np.expand_dims(img, axis=0)

    return img


def crop_image(img, img_bg):
    # Gray
    img_bg_gray = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate Difference
    diff_gray = cv2.absdiff(img_bg_gray, img_gray)
    # Diff Blur
    diff_gray_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)

    # find otsu's threshold value with OpenCV function
    ret, img_tresh = cv2.threshold(diff_gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # let's now draw the contour
    arr_cnt, a1 = cv2.findContours(img_tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        # Flag as edge_noise if the object is at a Corner
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
        if ca > 1300:

            # DISCARD as noise if W/H ratio > 7 to 1 (1x6 plate is 700px to 100px)
            # the conveyor belt has a join line that sometimes is detected as a contour, this ignores it
            if aspect_ratio <= 6:

                # DISCARD if at the Edge
                if not edge_noise:
                    validcontours.append(contour_index)

    # copy the original picture
    img_with_contours = img.copy()

    # call out if more than 1 valid contour is found
    if len(validcontours) > 1:
        print("There is more than 1 object in the picture")
    else:
        if len(validcontours) == 1:
            print("One object detected")
        else:
            print("No objects detected")
            # FYI: code below will most likely error out as it tries to iterate on an array

    # it might be possible we have more than 1 valid contour, iterating through them here
    # if there is zero contours, this most likely will error out
    for i in validcontours:
        cv2.drawContours(img_with_contours, arr_cnt, validcontours[i], (0, 255, 0), 3)

    # Display a Bounding Rectangle
    img_with_rectangle = img.copy()
    cropped_image = None
    for i in validcontours:
        x, y, w, h = cv2.boundingRect(arr_cnt[i])
        cropped_image = img_with_rectangle[y-100:y+h+100, x-100:x+w+100]
    return cropped_image


def main():
    # Load image
    base_path = '../../lego_dataset_orig/Cropped Images/'
    img_bg = cv2.imread('../../lego_dataset/background_backlit_A.jpg')
    img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))
    img_path = img_list[random.randrange(len(img_list))]
    img = cv2.imread(img_path)

    # plt.imshow(img.astype('uint8'))
    # plt.show()

    # Load model
    model = load_model('./models/model-80')

    # img_cropped = crop_image(img, img_bg)
    img_np = keras.preprocessing.image.img_to_array(img)
    img_processed = prep_img(img_np)

    predictions = model.predict(img_processed)
    score = predictions[0]
    classification = classes.get(np.argmax(score))
    print(img_path.split('\\')[-2])
    print(classification)


if __name__ == '__main__':
    main()

