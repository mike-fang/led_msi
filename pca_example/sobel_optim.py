import cv2
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
import numpy as np
from scipy.optimize import minimize

def norm_img(img):
    temp = img
    temp -= temp.min()
    temp /= temp.max()
    return temp
def get_sobel(img, ksize=5):
    Sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    Sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    return Sx, Sy

class OptimEdge:
    def __init__(self, raw_img, ksize=5, orthonormal=False):
        self.raw_img = raw_img
        self.ksize = ksize
        self.H, self.W, self.C = raw_img.shape

        img_flat = raw_img.reshape((-1, self.C))
        if orthonormal:
            # Transform to orthonormal channels
            self.pca = PCA(whiten=True)
            self.channels = self.pca.fit_transform(img_flat).reshape(raw_img.shape)
        else:
            self.channels = raw_img
        self.ONSx, self.ONSy = get_sobel(self.channels, ksize=ksize)
    def mean_S2(self, a):
        a /= (a**2).sum()**0.5
        Sx = self.ONSx @ a
        Sy = self.ONSy @ a
        return (Sx**2 + Sy**2).mean()
    def mean_S(self, a):
        a /= (a**2).sum()**0.5
        Sx = self.ONSx @ a
        Sy = self.ONSy @ a
        return ((Sx**2 + Sy**2)**0.5).mean()
    def compose(self, a):
        a /= (a**2).sum()**0.5
        img = self.channels @ a
        raw_mean = self.raw_img.mean(axis=2)
        if (raw_mean * img).mean() < 0:
            img *= -1
        return img
    def optimize(self, fn='S2'):
        cons = {'type': 'eq', 
                'fun': lambda x: (x**2).sum()**0.5 - 1
                }
        a0 = np.random.randn(3)
        a0 /= (a0**2).sum()**0.5
        if fn == 'S2':
            def f(a):
                return -self.mean_S2(a)
        elif fn == 'S1':
            def f(a):
                return -self.mean_S(a)
        res = minimize(f, a0, method='SLSQP', constraints=cons)
        return res
    def optim_compose(self, fn='S2', normalize=True):
        res = self.optimize(fn=fn)
        a = res['x']
        f = res['fun']

        print(f, a)
        img = self.compose(a)
        if normalize:
            img = norm_img(img)
        return img


if __name__ == '__main__':
    raw_img = cv2.imread('./shrimp.jpg')[:, :, ::-1]
    optim = OptimEdge(raw_img)
    s2_img = optim.optim_compose()
    s_img = optim.optim_compose(fn='S1')
    plt.subplot(221)
    plt.title('rgb')
    plt.imshow(raw_img)

    plt.subplot(222)
    plt.title('mean')
    plt.imshow(norm_img(raw_img.mean(axis=2)), cmap='gray')

    plt.subplot(223)
    plt.title('s2')
    plt.imshow(s2_img, cmap='gray')

    plt.subplot(224)
    plt.title('s1')
    plt.imshow(s_img, cmap='gray')
    plt.show()
