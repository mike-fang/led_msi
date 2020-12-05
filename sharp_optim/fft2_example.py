import cv2
import numpy as np
import matplotlib.pylab as plt

img = cv2.imread('./imgs/duck.jpg').mean(axis=2)
F_img = np.fft.fft2(img)
F_img = np.abs(F_img)
F_img = np.fft.fftshift(F_img)

plt.imshow(img)
plt.figure()
plt.imshow(np.log(F_img))
plt.show()
