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


class Model(abc.ABC, mp.Process):
    def __init__(self):
        self.input_queue = None
        self.output_queue = None

    def connect(self, input_queue, output_queue):
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        while True:
            if not self.queue_in.empty():
                data = self.queue_in.get()

                if data is None:
                    break

                output = self.process(data)
                self.output_queue.put(output)

    def process(self, input_data: Container):
        raise NotImplementedError


class Task(abc.ABC):
    pass
