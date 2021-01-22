import cv2
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import minimize

def norm_img(img):
    temp = img
    temp -= temp.min()
    temp /= temp.max()
    return temp
def get_sorbel(img, ksize=5):
    Sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    Sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    return Sx, Sy

def cb(x):
    return 
    print(x, np.linalg.norm(x))

class OptimEdge:
    def __init__(self, raw_img, norm_a=2, ksize=5, orthonormal=False):
        self.norm_a_ord = norm_a
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
        self.ONSx, self.ONSy = get_sorbel(self.channels, ksize=ksize)
    def a_norm(self, a):
        return np.linalg.norm(a, ord=self.norm_a_ord)
    def mean_S2(self, a):
        a /= self.a_norm(a)
        Sx = self.ONSx @ a
        Sy = self.ONSy @ a
        return (Sx**2 + Sy**2).mean()
    def mean_S(self, a):
        a /= self.a_norm(a)
        Sx = self.ONSx @ a
        Sy = self.ONSy @ a
        return ((Sx**2 + Sy**2)**0.5).mean()
    def compose(self, a):
        a /= self.a_norm(a)
        img = self.channels @ a
        raw_mean = self.raw_img.mean(axis=2)
        if (raw_mean * img).mean() < 0:
            img *= -1
        return img
    def optimize(self, fn='S2', pos=False):
        cons = {'type': 'eq', 
                'fun': lambda x: self.a_norm(x)**2 - 1
                }
        a0 = np.random.randn(3)
        a0 /= self.a_norm(a0)
        if pos:
            bounds = [(0, None), ] * 3
            a0[a0<0] *= -1
        else:
            bounds = None
        if fn == 'S2':
            def f(a):
                return -self.mean_S2(a)
        elif fn == 'S1':
            def f(a):
                return -self.mean_S(a)
        res = minimize(f, a0, method='SLSQP', jac=False, constraints=cons, callback=cb, bounds=bounds)
        return res
    def optim_compose(self, normalize=True, **kwargs):
        res = self.optimize(**kwargs)
        a = res['x']
        f = res['fun']

        print(f, a)
        img = self.compose(a)
        if normalize:
            img = norm_img(img)
        return img

if __name__ == '__main__':
    raw_img = cv2.imread('./shrimp.jpg')[:, :, ::-1]
    plt.subplot(321)
    plt.title('rgb')
    plt.imshow(raw_img)

    plt.subplot(322)
    plt.title('mean')
    plt.imshow(norm_img(raw_img.mean(axis=2)), cmap='gray')

    optim = OptimEdge(raw_img, norm_a=2)
    s2_img = optim.optim_compose(pos=True)
    s_img = optim.optim_compose(fn='S1', pos=True)
    plt.subplot(323)
    plt.title('s2; |a|_2=1')
    plt.imshow(s2_img, cmap='gray')

    plt.subplot(324)
    plt.title('s1; |a|_2=1')
    plt.imshow(s_img, cmap='gray')

    optim = OptimEdge(raw_img, norm_a=1)
    s2_img = optim.optim_compose(pos=True)
    s_img = optim.optim_compose(fn='S1', pos=True)
    plt.subplot(325)
    plt.title('s2; |a|_1=1')
    plt.imshow(s2_img, cmap='gray')

    plt.subplot(326)
    plt.title('s1; |a|_1=1')
    plt.imshow(s_img, cmap='gray')
    plt.show()
