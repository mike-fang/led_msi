import cv2
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
imgs_dir = os.path.join(curr_dir, 'imgs')

cam = cv2.VideoCapture(1)
img_counter = 0

# width, height = cam.get(3), cam.get(4)
# print(width, height)

# cam.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height * 2)

# width, height = cam.get(3), cam.get(4)
# print(width, height)


while cam.isOpened() and img_counter < 20:
    ret, frame = cam.read()
    if not ret:
        print(ret)
        break

    img_name = os.path.join(imgs_dir, f'opencv_frame_{img_counter}.jpg')
    cv2.imwrite(img_name, frame)
    print(f'{img_name} written!')
    img_counter += 1
    print(img_counter)

cam.release()
cv2.destroyAllWindows()

