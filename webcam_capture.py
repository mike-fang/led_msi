import cv2
import sys
import numpy as np
from time import sleep
import os
from relay_ft245r import relay_ft245r

def stream(get_image, args={}):
    while True:
        if args:
            img = get_image(**args)
        else:
            img = get_image()
        cv2.imshow("", img)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break


if __name__ == '__main__':
    args = {"-w": 1}

    i = 1
    while i < len(sys.argv):
        args[sys.argv[i]] = sys.argv[i + 1]
        i += 2

    webcam = cv2.VideoCapture(args["-w"])

    stream(lambda: webcam.read()[1])