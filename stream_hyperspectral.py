from pca_example.Lab_pca import *
from webcam_capture import *
from hyperspectral_capture import *

def calibrate():
    for i in range(4, 9):
        capture_image(i, array)
    return get_pca_fit(array)

def capture(pca):
    for i in range(4, 9):
        capture_image(i, array)
    return load_images(array, pca)

pca = calibrate()
stream(capture, pca)