import cv2
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import minimize

raw_img = cv2.imread('./imgs/shrimp.jpg')[:, :, ::-1]

def compose(hd_img, comps):
    comps /= np.linalg.norm(comps, ord=2)
    return hd_img @ comps

def sobel_sharpness(img):
    Sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    Sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return np.mean((Sx**2 + Sy**2)**0.5)

def sobel_sharpness_sq(img):
    Sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    Sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    return np.mean((Sx**2 + Sy**2))

def gabor_sharpness(img):
    Gx = ...
    Gy = ...
    return np.mean((Gx**2 + Gy**2)**0.5)

def obj_4(img):
    return ...

def optim_comps(hd_img, sharp_fn):
    def f(comps):
        img = compose(hd_img, comps)
        return -sharp_fn(img)

    a0 = np.random.randn(3)
    a0 /= np.linalg.norm(a0, ord=2)

    cons = {'type' : 'eq',
            'fun' : lambda x: np.linalg.norm(x)**2 - 1
            }
    res = minimize(f, a0, method='SLSQP', jac=False, constraints=cons)
    comps = res['x']
    if np.sum(comps) < 0:
        comps *= -1
    return comps

def optim_img(hd_img, sharp_fn):
    comps = optim_comps(hd_img, sharp_fn)
    return compose(hd_img, comps)


img = optim_img(raw_img, sobel_sharpness)
plt.imshow(img, cmap='gray')
plt.figure()
img = optim_img(raw_img, sobel_sharpness_sq)
plt.imshow(img, cmap='gray')
plt.show()
