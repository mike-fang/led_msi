from relay_ft245r import relay_ft245r
import sys
import time
import numpy as np
import cv2
import os
from matplotlib import image
from matplotlib import pyplot

curr_dir = os.path.dirname(os.path.abspath(__file__))
imgs_dir = os.path.join(curr_dir, 'imgs')

#plugged into USB port 1
webcam = cv2.VideoCapture(1)
colors = ('z_blank', 'zz_blank', 'zzz_blank', 'yellow', 'white', 'blue', 'red', 'green')

rb = relay_ft245r.FT245R()
dev_list = rb.list_dev()
dev = dev_list[0]
rb.connect(dev)

for i in range(1, 9):
    rb.switchon(i)
    time.sleep(.5)
    ret, frame = webcam.read()
    if not ret:
        print(ret)
        break

    f_name = os.path.join(curr_dir, 'imgs', f'frame_{colors[i - 1]}.png')
    cv2.imwrite(f_name, frame)
    time.sleep(.5)
    rb.switchoff(i)

webcam.release()
cv2.destroyAllWindows()


