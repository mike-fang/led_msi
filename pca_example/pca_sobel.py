import numpy as np
import matplotlib.pylab as plt
import pickle
from sklearn.decomposition import PCA
from skimage.color import lab2rgb
from Lab_pca import lab_pca_fitted
from sobel_optim import get_sobel, norm_img
import cv2

from scipy.optimize import minimize


class ComponentsOptimEdge:
    def __init__(self, ms_img):
        self.ms_img = ms_img
        self.channels = ms_img.shape[2]
        self.sobel_x, self.sobel_y= get_sobel(ms_img)

    def magnitude(self, components):
        components /= (components ** 2).sum() ** 0.5
        Sx = self.sobel_x @ components
        Sy = self.sobel_y @ components
        return (Sx**2 + Sy**2).mean()

    def optimize(self):
        cons = {'type': 'eq',
                'fun': lambda x: (np.abs(x)).sum() - 1
                }
        a0 = np.random.randn(self.channels)
        a0 /= (a0**2).sum()**0.5
        def f(a):
            return -self.magnitude(a)
        res = minimize(f, a0, method='SLSQP', constraints=cons)
        print(res)
        return res


if __name__ == '__main__':

    #ms_img = np.load("../images_data/eraser.npy")[:, :, 9:]

    ms_img = cv2.imread("../images_data/shrimp.jpg")[:,:,::-1]
    # for i in range(5):
    #     plt.subplot(2, 4, i + 1)
    #     plt.imshow(ms_img[:, :, i * 3:(i + 1) * 3])
    # plt.show()
    #
    # with open("pca_model", "rb") as f:
    #     pca_loaded = pickle.load(f)
    #
    # pca_img = lab_pca_fitted(pca=pca_loaded, ms_img=ms_img)
    # plt.imshow(pca_img)
    # plt.show()

    comp_optimize = ComponentsOptimEdge(ms_img)
    components = comp_optimize.optimize()['x']
    print(components)
    components /= (components ** 2).sum() ** 0.5
    if components.sum() < 0:
        components *= -1
    img = ms_img @ components

    plt.imshow(img, cmap="gray")
    plt.show()
