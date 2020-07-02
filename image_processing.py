from matplotlib import image
from matplotlib import pyplot
from PIL import Image
import os

images = []
curr_dir = os.path.dirname(os.path.abspath(__file__))
imgs_dir = os.path.join(curr_dir, 'imgs')
colors = ('z_blank', 'zz_blank', 'zzz_blank', 'yellow', 'white', 'blue', 'red', 'green')


for i in range(1, 9):
    img_path = os.path.join(curr_dir, 'imgs', f'frame_{colors[i - 1]}.png')
    img = Image.open(img_path)
    images.append(img)

    print(img.format)
    print(img.size)
    print(image.mode)
    # display the array of pixels as an image
    load_image.show()