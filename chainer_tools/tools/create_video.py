import numpy as np
from skimage.io import imread
import cv2
import os


def create_video_from_files(img_fns, out_file, delete_imgs=False):
    dir_name = os.path.split(out_file)[0]
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    img_fns.sort()
    imgs = []
    for img_fn in img_fns:
        imgs.append(cv2.imread(img_fn))
    create_video(imgs, out_file)

    if delete_imgs:
        for img_fn in img_fns:
            os.remove(img_fn)


def create_video(imgs, out_file):
    """
    Images needs to be [0, 255] scale and BGR format.

    Format of the vidoe should be .avi
    """
    height, width, _ = imgs[0].shape
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    # In case X264 doesn't work\n",
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 30.0
    video = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    for img in imgs:
        video.write(img.astype(np.uint8))

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    imgs = []
    for i in range(50):
        # imgs.append(np.random.uniform(high=255,size=(256, 256, 3)))
        imgs.append(np.broadcast_to(np.array([255, 0, 0]), (256, 256, 3)))
    create_video(imgs, 'a.avi')
