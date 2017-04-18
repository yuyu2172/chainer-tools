import chainercv
import copy

import numpy as np


def _copy_transform(in_data):
    out = []
    for elem in in_data:
        if isinstance(elem, np.ndarray):
            elem = elem.copy()
        else:
            elem = copy.copy(elem)
        out.append(elem)
    return tuple(out)


def copy_dataset(dataset):
    return chainercv.datasets.TransformDataset(dataset, _copy_transform)
