import numpy as np
import cv2
import matplotlib.pylab as plt
from skimage.filters import gabor_kernel
from scipy.signal import convolve


def norm_img(img):
    temp = np.array(img, dtype=np.float)
    temp -= temp.min()
    temp /= (temp.max() + 1e-5)
    return temp
def imshow(img, norm=False):
    if norm:
        img_ = norm_img(img)
    else:
        img_ = img
    plt.imshow(img_, cmap='gray')
def plot_conv(kernel):
    conv = convolve(kernel, img_gray, mode='valid')
    plt.figure(figsize=(6, 12))
    plt.subplot(311)
    imshow(kernel)
    plt.subplot(312)
    imshow(conv)
    plt.subplot(313)
    imshow(conv**2)

img = cv2.imread('./desert.jpg')
img_gray = img.mean(axis=2)

Sx = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
    ])
Sy = Sx.T

kernels = [Sx, Sy]
if True:
    sigma = 10
    kernels = []
    for freq in [0.05, 0.2]:
        for theta in range(4):
            theta = theta / 4 * np.pi
            K = np.real(gabor_kernel(freq, theta=theta, sigma_x=10, sigma_y=10))
            kernels.append(K)

for k in kernels:
    plot_conv(k)

plt.figure()
imshow(img_gray)
plt.show() 
