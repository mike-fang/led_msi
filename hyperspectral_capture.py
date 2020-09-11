from relay_ft245r import relay_ft245r
import sys
import time
import numpy as np
import cv2
import os
from matplotlib import image
from matplotlib import pyplot as plt
from timeit import default_timer as timer

def capture_image(led, ms_img, rb, webcam, setup_time=0, sleep_time=0.08, show=False, path=None):
    start = timer()
    time_elapsed = 0
    while time_elapsed < setup_time:
        ret, frame = webcam.read()
        if show:
            cv2.imshow("", frame)
        time_elapsed = timer() - start
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    rb.switchon(led)

    time.sleep(sleep_time)
    ret, frame = webcam.read()
    if not ret:
        print(ret)
        return
    if path:
        cv2.imwrite(path, frame)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if show:
        print("image shape: ", image.shape)
    H, W, C = image.shape
    ms_img[:, :, (led - 1) * C:led * C] = image
    rb.switchoff(led)

def initialize_rb():
    rb = relay_ft245r.FT245R()
    dev_list = rb.list_dev()
    dev = dev_list[0]
    rb.connect(dev)
    return rb

def empty_sized_array(webcam):
    ret, frame = webcam.read()
    H, W, C = frame.shape
    array = np.zeros((H, W, C * 8), dtype=np.uint8)
    return array

if __name__ == '__main__':

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    colors = ('z_blank', 'zz_blank', 'zzz_blank', 'yellow', 'white', 'blue', 'red', 'green')

    args = {"-setup": 15, "-sleep": 0.01, "-show": True, "-w": 1, "-path": os.path.join(curr_dir, 'imgs'), "-test":False}

    i = 1
    while i < len(sys.argv):
        args[sys.argv[i]] = sys.argv[i + 1]
        i += 2

    rb = initialize_rb()
    webcam = cv2.VideoCapture(int(args["-w"]))
    array = empty_sized_array(webcam)

    for i in range(4, 9):
        f_name = os.path.join(args["-path"], f'frame_{colors[i - 1]}.png')
        capture_image(i, array, rb, webcam, setup_time=float(args["-setup"]), sleep_time=float(args["-sleep"]), show=bool(args["-show"]), path=f_name)
        args["-setup"] = min(int(args["-setup"]), 1)

    webcam.release()
    cv2.destroyAllWindows()

    if bool(args["-show"]):
        for i in range(8):
            plt.subplot(2,4,i+1)
            plt.imshow(array[:,:,i*3:(i+1)*3])
        plt.show()

    np_path = os.path.join(curr_dir, 'pca_example', 'led_imgs')
    np.save(np_path, array)
