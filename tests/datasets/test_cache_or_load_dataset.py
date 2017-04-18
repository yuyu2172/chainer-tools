import unittest

import numpy as np
import os
import tempfile

from chainer.datasets import TupleDataset

from chainer_tools.datasets.cache_or_load_dataset import cache_or_load_dataset


class TestCacheOrLoadDataset(unittest.TestCase):

    def test_cache_or_load_dataset(self):
        N = 10
        float_dataset = np.random.uniform(size=(N, 3, 10, 10)).astype(np.float32)
        int_dataset = np.random.uniform(size=(N,)).astype(np.int32)
        dataset = TupleDataset(float_dataset, int_dataset)

        temp_dir = tempfile.mkdtemp()
        fn = os.path.join(temp_dir, 'cache.dat')
        obtained = cache_or_load_dataset(fn, dataset)

        val_1, val_2 = obtained[1]
        expected_1, expected_2 = dataset[1]

        np.testing.assert_equal(val_1, expected_1)
        np.testing.assert_equal(val_2, expected_2)

        # test loading
        dataset = cache_or_load_dataset(fn)
        self.assertEqual(len(dataset), N)
        for i in range(N):
            expected_1, expected_2 = dataset[i]
            val_1, val_2 = obtained[i]
            np.testing.assert_equal(val_1, expected_1)
            np.testing.assert_equal(val_2, expected_2)

    def test_cache_or_load_dataset_not_tuple(self):
        N = 10
        float_dataset = np.random.uniform(size=(N, 3, 10, 10)).astype(np.float32)

        temp_dir = tempfile.mkdtemp()
        fn = os.path.join(temp_dir, 'cache.dat')
        obtained = cache_or_load_dataset(fn, float_dataset)

        val_1 = obtained[1]
        expected_1 = float_dataset[1]

        np.testing.assert_equal(val_1, expected_1)

        # test loading
        dataset = cache_or_load_dataset(fn)
        expected_1 = dataset[1]
        np.testing.assert_equal(val_1, expected_1)
        self.assertEqual(len(dataset), N)
