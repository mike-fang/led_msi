from pca_example.Lab_pca import *
from webcam_capture import *
from hyperspectral_capture import *


def calibrate(sized_array, rb, webcam):
    for i in range(4, 9):
        capture_image(i, sized_array, rb, webcam)
    return get_pca_fit(sized_array)

def capture(pca, sized_array, rb, webcam):
    for i in range(4, 9):
        capture_image(i, sized_array, rb, webcam)
    return load_images(sized_array, pca)

if __name__ == '__main__':

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    colors = ('z_blank', 'zz_blank', 'zzz_blank', 'yellow', 'white', 'blue', 'red', 'green')

    args = {"-w": 1}

    i = 1
    while i < len(sys.argv):
        args[sys.argv[i]] = sys.argv[i + 1]
        i += 2

    webcam = cv2.VideoCapture(args["-w"])
    array = empty_sized_array(webcam)

    rb = initialize_rb()
    pca = calibrate(array, rb, webcam)
    stream(capture, args={"pca":pca, "rb": rb, "sized_array":array, "webcam":webcam})