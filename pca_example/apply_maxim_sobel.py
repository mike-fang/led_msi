from scipy.signal import convolve
import numpy as np
import matplotlib.pylab as plt
from sobel_optim import get_sobel, norm_img
import cv2
from skimage.filters import gabor_kernel
from scipy.optimize import minimize
from time import time


class ComponentsOptimEdge:

    def __init__(self, ms_img, obj_fn):
        sobel_fns = {"sobel": self.sobel_magnitude, "gabor":self.gabor_obj_fn}
        self.ms_img = ms_img
        self.Sx, self.Sy = get_sobel(ms_img)
        self.channels = ms_img.shape[2]
        self.obj_fn = sobel_fns[obj_fn]

    def optimize(self):
        cons = {'type': 'eq',
                'fun': lambda x: (np.abs(x)).sum() - 1
                }
        a0 = np.random.randn(self.channels)
        self.normalize(a0)
        res = minimize(lambda x : -1 * self.obj_fn(x), a0, method='SLSQP', constraints=cons)
        return res

    def sobel_magnitude(self, components):
        # t0 = time()
        self.normalize(components)
        # img = self.ms_img @ components
        # Sx, Sy = get_sobel(img)
        t0 = time()
        Sx = self.Sx @ components
        Sy = self.Sy @ components
        print(time() - t0)
        S2 = np.mean((Sx**2 + Sy**2) ** 0.5)
        # print(time() - t0)
        return S2

    def gabor_obj_fn(self, components):
        self.normalize(components)
        img = self.ms_img @ components
        kernel_x = gabor_kernel(1, theta=0)
        kernel_y = gabor_kernel(1, theta=np.pi / 2)
        Gx = np.real(convolve(kernel_x, img))
        Gy = np.real(convolve(kernel_y, img))
        return np.mean((Gx ** 2 + Gy ** 2) ** 0.5)

    def normalize(self, components):
        components /= (components ** 2).sum() ** 0.5




if __name__ == '__main__':

    ms_img = np.load("../images_data/eraser.npy")[:,:,12:24]
    print(get_sobel(ms_img)[0].shape)

    print(ms_img.shape)
    comp_optimize = ComponentsOptimEdge(ms_img, "sobel")
    components = comp_optimize.optimize()['x']
    print(components)
    components /= (components ** 2).sum() ** 0.5
    if components.sum() < 0:
        components *= -1

    #plt.imshow(ms_img, cmap="gray")
    #plt.show()

    img = ms_img @ components

    plt.imshow(img, cmap="gray")
    plt.show()
