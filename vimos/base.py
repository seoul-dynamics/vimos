import abc
import os
import os.path as osp
import time

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

    def apply(self, func):
        self.data = func(self).data
        return self

    def _prepare_data(self, input_source):
        raise NotImplementedError


class Editor(abc.ABC):
    def __call__(self, image):
        return image.apply(self._process)


class Modifier(abc.ABC):
    def __call__(self, skeleton):
        if isinstance(skeleton, list):
            return [self._process(s) for s in skeleton]
        return skeleton.apply(self._process)


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

                output = self.predict(data)
                self.queue_out.put(output)

    def predict(self, input_data: Container):
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
    def __init__(self, task_config):
        self.task_config = task_config

        self.queue_in = None
        self.queue_out = None
        self.cache = None

    def start(self):
        pass

    def process(self, input_data):
        pass

    def _apply(self, input_data):
        if self.task_config.editor is not None:
            input_data = self.task_config.editor(input_data)

        while self.queue_in.qsize() > 10:
            try:
                temp = self.queue_in.get_nowait()
                del temp
            except:
                break
        self.queue_in.put(input_data)

        updated = True

        if self.cache is None:
            output_data = self.queue_out.get()
            self.cache = output_data
        else:
            try:
                output_data = self.queue_out.get_nowait()
                self.cache = output_data
            except Exception:
                output_data = self.cache
                updated = False

        if self.task_config.modifier is not None:
            output_data = self.task_config.modifier(output_data)

        return output_data, updated
