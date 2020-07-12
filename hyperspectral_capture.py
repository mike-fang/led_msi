from relay_ft245r import relay_ft245r
import sys
import time
import numpy as np
import cv2
import os
from matplotlib import image
from matplotlib import pyplot as plt
from timeit import default_timer as timer

curr_dir = os.path.dirname(os.path.abspath(__file__))
imgs_dir = os.path.join(curr_dir, 'imgs')

#plugged into USB port 1
webcam = cv2.VideoCapture(1)
colors = ('z_blank', 'zz_blank', 'zzz_blank', 'yellow', 'white', 'blue', 'red', 'green')
images = []

rb = relay_ft245r.FT245R()
dev_list = rb.list_dev()
dev = dev_list[0]
rb.connect(dev)

ret,frame = webcam.read()
H,W,C = frame.shape
array = np.zeros((H,W,C*8), dtype=np.uint8)

for i in range(1, 9):

    start = timer()
    time_elapsed = 0
    setup_time = 15 if i == 1 else 3
    while time_elapsed < setup_time:
        ret, frame = webcam.read()
        cv2.imshow("", frame)
        time_elapsed = timer() - start
        if cv2.waitKey(1) & 0xff == ord("q"):
            break


    rb.switchon(i)
    time.sleep(.5)
    ret, frame = webcam.read()
    if not ret:
        print(ret)
        break

    f_name = os.path.join(curr_dir, 'imgs', f'frame_{colors[i - 1]}.png')
    time.sleep(.5)
    cv2.imwrite(f_name, frame)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("image shape: ", image.shape)
    array[:,:,(i - 1)*C:i*C] = image
    images.append(image)
    rb.switchoff(i)

webcam.release()
cv2.destroyAllWindows()

for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(array[:,:,i*3:(i+1)*3])

plt.show()

# hyperspectral_image = cv2.fromarray(array)
# f_name = os.path.join(curr_dir, 'imgs', f'frame_{hyperspectral_image}.png')
# cv2.imwrite(f_name, hyperspectral_image)