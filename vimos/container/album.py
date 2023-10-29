from typing import List, Union
import multiprocessing as mp
import cv2
import numpy as np

from vimos.container import Container
from vimos.editor import Editor


class Album(Container):
    def _prepare_data(self, input_source: List[Union[str, np.ndarray]]):
        if isinstance(input_source[0], str):
            input_source = [cv2.imread(src) for src in input_source]
        return input_source

    def apply(self, editor: Editor, inplace: bool = False):
        if inplace:
            with mp.Pool() as pool:
                self.data = pool.map(editor, self.data)
            return self
        else:
            with mp.Pool() as pool:
                data = pool.map(editor, self.data.copy())
            return Album(data)
