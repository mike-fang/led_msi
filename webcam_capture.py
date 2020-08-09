import cv2
import numpy as np
from time import sleep
import os
from relay_ft245r import relay_ft245r


# rb = relay_ft245r.FT245R()
# dev_list = rb.list_dev()
# dev = dev_list[0]
# rb.connect(dev)
#
# curr_dir = os.path.dirname(os.path.abspath(__file__))
# imgs_dir = os.path.join(curr_dir, 'imgs')
# if not os.path.isdir(imgs_dir):
#     print(f'Making directory {imgs_dir}')
#     os.mkdir(imgs_dir)

webcam = cv2.VideoCapture(1)


def stream(get_image, pca=None):
    while True:
        if pca:
            img = get_image(pca)
        else:
            img = get_image()
        cv2.imshow("", img)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break

        # f_name = os.path.join(curr_dir, 'imgs', f'frame_{n}.png')
        # print(f'Saving image to {f_name}')
        # cv2.imwrite(f_name, frame)
        # sleep(.5)

if __name__ == '__main__':
    stream(lambda: webcam.read()[1])