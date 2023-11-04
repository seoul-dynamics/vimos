import abc
import os
import os.path as osp

import numpy as np


class Container(abc.ABC):
    def __init__(self, input_source):
        self.data = self._prepare_data(input_source)

    def save(self, path: str):
        if not osp.exists(osp.dirname(path)):
            os.makedirs(osp.dirname(path))
        np.save(path, self.data)

    def load(self, path: str):
        self.data = np.load(path)

    def apply(self, editor, inplace=False):
        raise NotImplementedError

    def _prepare_data(self, input_source):
        raise NotImplementedError


class Editor(abc.ABC):
    pass


class Modifier(abc.ABC):
    pass


class Model(abc.ABC):
    pass


class Task(abc.ABC):
    pass
