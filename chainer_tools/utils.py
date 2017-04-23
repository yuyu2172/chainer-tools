import numpy as np


def figure2array(f):
    f.canvas.draw()
    arr = np.fromstring(
        f.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    arr = arr.reshape(
        f.canvas.get_width_height()[::-1] + (3,))

    # output CHW
    arr = arr.transpose(2, 0, 1)
    return arr
