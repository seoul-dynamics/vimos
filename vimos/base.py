import abc
import os
import os.path as osp
import multiprocessing as mp

import torch
import numpy as np


class Container(abc.ABC):
    def __init__(self, input_source=None):
        if input_source is not None:
            self.data = self._prepare_data(input_source)

    def save(self, path: str):
        if len(path.split("/")) > 1 and not osp.exists(osp.dirname(path)):
            os.makedirs(osp.dirname(path))
        np.save(path, self.data)

    def load(self, path: str):
        self.data = np.load(path)
        return self

    def apply(self, editor, inplace=False):
        raise NotImplementedError

    def _prepare_data(self, input_source):
        raise NotImplementedError


class Editor(abc.ABC):
    pass


class Modifier(abc.ABC):
    pass


class Model(abc.ABC):
    def __init__(self):
        super().__init__()

        self.queue_in = None
        self.queue_out = None

    def connect(self, queue_in, queue_out):
        self.queue_in = queue_in
        self.queue_out = queue_out

    def run(self):
        while True:
            if not self.queue_in.empty():
                data = self.queue_in.get()

                if data is None:
                    break

                output = self.process(data)
                self.queue_out.put(output)

    def process(self, input_data: Container):
        raise NotImplementedError


class Metric(abc.ABC):
    similar_small = None

    def __call__(self, ref_skeleton, query_skeleton):
        pass

    def _evaluate(self, ref_skeleton, query_skeleton):
        pass

    def _normalize(self, ref_vector, query_vector):
        pass


class Task(abc.ABC):
    pass
