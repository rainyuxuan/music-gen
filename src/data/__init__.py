"""
Download or generate data.
"""

from dataclasses import dataclass

from .dataset import *


# from loader import *


@dataclass
class Datasets:
    train: Dataset
    val: Dataset
    test: Dataset


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
