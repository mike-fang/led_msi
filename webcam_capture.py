import cv2
import numpy as np
from time import sleep
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
imgs_dir = os.path.join(curr_dir, 'imgs')
if not os.path.isdir(imgs_dir):
    print(f'Making directory {imgs_dir}')
    os.mkdir(imgs_dir)

webcam = cv2.VideoCapture(0)
for n in range(5):
    ret, frame = webcam.read()
    f_name = os.path.join(curr_dir, 'imgs', f'frame_{n}.png')
    print(f'Saving image to {f_name}')
    cv2.imwrite(f_name, frame)
    sleep(.5)
