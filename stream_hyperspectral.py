from pca_example.Lab_pca import *
from webcam_capture import *
from hyperspectral_capture import *

def calibrate():
    for i in range(4, 9):
        capture_image(i, array)
    return load_images(array, False)

def capture(pca):
    for i in range(4, 9):
        capture_image(i, array)
    return load_images(array, pca)

pca = get_pca_fit(calibrate())
stream(capture, pca)