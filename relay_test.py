from relay_ft245r import relay_ft245r
import sys
import time
import numpy as np
import cv2
import os

debug = False
colors = ('z_blank', 'zz_blank', 'zzz_blank', 'yellow', 'white', 'blue', 'red', 'green')

def get_intervals(periods, tmax):
    intervals = []
    for n, T in enumerate(periods):
        for t in np.arange(0, tmax, T):
            intervals.append((n, t))
    intervals.sort(key=lambda x:x[1])
    return intervals, get_diff(intervals)
def get_diff(intervals):
    n, t0 = intervals[0]
    diff = [(n, 0)]
    for (n, t) in intervals[1:]:
        dt = t - t0
        diff.append((n, dt))
        t0 = t
    return diff

def test(rb):
    rb.switchon(3)
    time.sleep(1)
    rb.switchoff(3)

curr_dir = os.path.dirname(os.path.abspath(__file__))
imgs_dir = os.path.join(curr_dir, 'imgs')
if not os.path.isdir(imgs_dir):
    print(f'Making directory {imgs_dir}')
    os.mkdir(imgs_dir)

webcam = cv2.VideoCapture(0)

if __name__ == '__main__':
    rb = relay_ft245r.FT245R()
    dev_list = rb.list_dev()

    # list of FT245R devices are returned
    if len(dev_list) == 0:
        print('No FT245R devices found')
        sys.exit()
        
    dev = dev_list[0]
    print('Using device with serial number ' + str(dev.serial_number))

    rb.connect(dev)

    if debug:
        test(rb)

    for i in range(1, 9):
        rb.switchon(i)
        time.sleep(.5)
        ret, frame = webcam.read()
        f_name = os.path.join(curr_dir, 'imgs', f'frame_{colors[i - 1]}.png')
        print(f'Saving image to {f_name}')
        cv2.imwrite(f_name, frame)
        time.sleep(.5)
        rb.switchoff(i)


    #Example code
    if debug:
        periods = (0.2, 0.3, 0.5, 0.7, 1.1, 1.3, 1.7, 1.9)
        _, diff = get_intervals(periods, 5)

        for n, dt in diff:
            time.sleep(dt)
            rb.toggle(n+1)

        rb.set_state(np.zeros(8))

        for n in range(8):
            time.sleep(.1)
            state = np.zeros(8)
            state[n] = 1
            rb.set_state(state)

        for n in range(64):
            time.sleep(1/(n+1))
            state = np.zeros(8)
            state[n*3 % 8] = 1
            rb.set_state(state)

        rb.set_state(np.zeros(8))
