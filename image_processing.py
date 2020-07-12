from matplotlib import image
from matplotlib import pyplot
from PIL import Image
import cv2
import os

images = []
curr_dir = os.path.dirname(os.path.abspath(__file__))
imgs_dir = os.path.join(curr_dir, 'imgs')
colors = ('z_blank', 'zz_blank', 'zzz_blank', 'yellow', 'white', 'blue', 'red', 'green')


for i in range(1, 9):
    img_path = os.path.join('imgs', f'frame_{colors[i - 1]}.png')
    im = cv2.imread(img_path)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    cv2.imwrite(os.path.join(imgs_dir, f'opencv_{colors[i - 1]}.png'), img)
    print (type(img))
    # display the array of pixels as an image