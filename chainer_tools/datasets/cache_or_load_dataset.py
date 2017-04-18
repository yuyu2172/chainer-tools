import chainer
import numpy as np
import os
import pickle


def cache_or_load_dataset(path, dataset=None):
    """Caches a dataset if it is not already cached, or loads it otherwise

    This caches a dataset at :obj:`path` if it has not been cached yet.
    The dataset is cached by creating a numpy memmapped file containing data.
    This loads a dataset cached at :obj:`path` if the file previously
    created by this function already exists.

    :obj:`dataset` is either a dataset object or :obj:`None`.
    If dataset is a dataset object, dataset returned is a
    instantiation of :class:`chainer.datasets.TupleDataset` regardless
    of existance of the cache in the filesystem.
    If :obj:`dataset` is :obj:`None` and cache can not be found,
    this function returns :obj:`None`. If :obj:`dataset` is :obj:`None` and
    cache can be found, the data loaded from the cache is returned.

    .. note::

        Data has to be same length across all examples

    Args:
        path (string): A path where dataset is cached or loaded from.
        dataset: A dataset object or :obj:`None`.

    Returns:
        a dataset or :obj:`None` depending on the argument :obj:`dataset`.
        The dataset is an instantiation of
        :class:`chainer.datasets.TupleDataset`. It loads data from a hdf5
        file locating at :obj:`path`.

    """
    if os.path.exists(path):
        return _load_cached_dataset(path)

    return _cache_dataset(dataset, path)


def _load_cached_dataset(path):
    with open(path, 'rb') as f:
        summary = pickle.load(f)

    dsets = []
    for i in range(summary['length_datum']):
        dset = np.memmap(path + '_{}'.format(i),
                         mode='r',
                         order='C',
                         shape=summary['shapes'][i],
                         dtype=summary['dtypes'][i])
        dsets.append(dset)

    if summary['is_datum_tuple']:
        dataset = chainer.datasets.TupleDataset(*dsets)
    else:
        dataset = dsets[0]
    return dataset


def _cache_dataset(dataset, path):
    if dataset is None:
        return

    datum = dataset[0]
    is_datum_tuple = True
    if not isinstance(datum, tuple):
        is_datum_tuple = False
        datum = (datum,)

    dsets = []
    shapes = []
    dtypes = []
    for i, value in enumerate(datum):
        dset = np.memmap(
            filename=path + '_{}'.format(i), mode='w+',
            order='C',
            shape=(len(dataset),) + value.shape,
            dtype=value.dtype)
        dsets.append(dset)
        shapes.append((len(dataset),) + value.shape)
        dtypes.append(value.dtype)

    for idx in range(len(dataset)):
        datum = dataset[idx]
        if not isinstance(datum, tuple):
            datum = (datum,)
        for i, val in enumerate(datum):
            dsets[i][idx] = val

    summary = {
        'is_datum_tuple': is_datum_tuple,
        'length_datum': len(datum),
        'shapes': shapes,
        'dtypes': dtypes
    }
    with open(path, 'wb') as f:
        pickle.dump(summary, f)

    if is_datum_tuple:
        dataset = chainer.datasets.TupleDataset(*dsets)
    else:
        dataset = dsets[0]
    return dataset


if __name__ == '__main__':
    dataset, _ = chainer.datasets.get_mnist()
    path = 'foo.hdf5'

    dataset = cache_or_load_dataset(path, dataset)
