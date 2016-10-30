#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension


setup(
    name='chainer_tools',
    version='0.0.1',
    packages=find_packages(),
    description='Tools to work with Chainer',
    author='Yusuke Niitani',
    author_email='yuyuniitani@gmail.com',
)
