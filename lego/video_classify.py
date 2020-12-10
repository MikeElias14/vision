import cv2
from urllib.request import urlopen
# need to use TF here as thats what the model used
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import classes


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

    # Find Contour
    arr_cnt, a1 = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        # TODO: Tweak this value, better camera, higher thresh
        if ca > 2500:
            print(f"Contour Area: {ca}")
            if not edge_noise:
                validcontours.append(contour_index)

    print(f"{len(validcontours)} objects detected")

    # Crop the img
    img_copy = img.copy()
    imgs_rects = []
    for i in validcontours:
        x, y, w, h = cv2.boundingRect(arr_cnt[i])
        # Make cropped image larger than bonding rectangle
        # 2x4 is 700x700,
        delta_y = 0
        delta_x = 0
        try:
            cropped_image = img_copy[y-delta_y:y+h+delta_y, x-delta_x:x+w+delta_x]
            imgs_rects.append([cropped_image, [x, y, x+w, y+h]])
        except Exception as e:
            print("Edge case")
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
    # Load Background image
    img_bg = cv2.imread('../../lego_dataset/my_images/background.jpg')
    img_bg = cv2.resize(img_bg, (1080, 1920), interpolation=cv2.INTER_NEAREST)

    # Load model
    model = load_model('./models/lego_CNN_small.h5')
    classifications = classes.classes_CNN_Small

    # define the target brick
    target = "Brick_2x2"
    target_key = list(classifications.keys())[list(classifications.values()).index(target)]

    url = 'http://192.168.0.114:8080/shot.jpg'

    while True:
        img_arr = np.array(bytearray(urlopen(url).read()), dtype=np.uint8)
        img_orig = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        imgs_rects = crop_image(img_orig, img_bg)
        class_rects = {}

        target_img = []
        target_confidence = 0

        imgs_to_predict = []
        rects = []

        # data of form [img, [x,y,x1,x2]]
        for img_rect in imgs_rects:
            try:
                img = tf.image.convert_image_dtype(img_rect[0], tf.float32)
                img_resized = tf.image.resize(img, [224, 224]).numpy()
                imgs_to_predict.append(img_resized)
                rects.append(img_rect[1])
            except Exception as e:
                print("Weirdness happened")

        # Get prediction
        if len(imgs_to_predict) != 0:

            print(f"predicting {len(imgs_to_predict)} images")
            imgs_to_predict = np.array(imgs_to_predict)
            predictions = model.predict(imgs_to_predict)

            for score in predictions:
                classification = classifications.get(np.argmax(score))
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
        cv2.imshow('IPWebcam', img_classed)

        # Quit if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

