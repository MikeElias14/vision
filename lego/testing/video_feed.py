from urllib.request import urlopen
import cv2
import numpy as np

URL = 'http://192.168.0.114:8080/shot.jpg'

while True:
    img_arr = np.array(bytearray(urlopen(URL).read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    cv2.imshow('IPWebcam', img)

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
