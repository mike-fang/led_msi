import cv2
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import minimize

raw_img = cv2.imread('./imgs/shrimp.jpg')[:, :, ::-1]

def compose(hd_img, components):
    components /= np.linalg.norm(components)
    return hd_img @ components

def sobel_sharpness(img):
    Sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    Sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return np.mean((Sx**2 + Sy**2)**0.5)
def sobel_sharpness_sq(img):
    Sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    Sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return np.mean((Sx**2 + Sy**2)**1)

def optimize(hd_img, sharp_fn):
    cons = {'type' : 'eq',
            'fun' : lambda x: np.linalg.norm(x)**2 - 1
            }
    def f(a):
        img = compose(hd_img, a)
        return -sharp_fn(img)
    a0 = np.random.randn(3)
    a0 /= np.linalg.norm(a0)**2
    res = minimize(f, a0, method='SLSQP', jac=False, constraints=cons)
    return res

def optim_img(hd_img, sharp_fn):
    a = optimize(hd_img, sharp_fn)['x']
    if a.sum() < 0:
        a *= -1
    print(a)
    return compose(hd_img, a)


img = optim_img(raw_img, sobel_sharpness)
plt.imshow(img, cmap='gray')
plt.figure()
img = optim_img(raw_img, sobel_sharpness_sq)
plt.imshow(img, cmap='gray')
plt.show()


