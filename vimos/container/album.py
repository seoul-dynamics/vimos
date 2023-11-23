from typing import List, Union, Callable
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from vimos.base import Container, Editor
from vimos.container import Photo


class Album(Container):
    def apply(self, editor_func: Callable):
        with mp.Pool() as pool:
            self.data = pool.map(editor_func, self.data)
            return self

    def save(self, path: str):
        self.data = np.stack([photo.data for photo in self.data])
        super().save(path)
        self.data = self._prepare_data(self.data)

    def load(self, path: str):
        super().load(path)
        self.data = [Photo(photo) for photo in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _prepare_data(
        self, input_source: Union[List[str], List[np.ndarray]], num_workers: int = 4
    ):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            data = list(executor.map(lambda src: Photo(src), input_source))
        return data
