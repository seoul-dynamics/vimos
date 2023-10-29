from typing import Union
import cv2
import numpy as np

from vimos.container import Container
from vimos.editor import Editor


class Photo(Container):
    def _prepare_data(self, input_source: Union[str, np.ndarray]):
        if isinstance(input_source, str):
            input_source = cv2.imread(input_source)
        return input_source

    def apply(self, editor: Editor, inplace: bool = False):
        if inplace:
            self.data = editor(self.data)
            return self
        else:
            return Photo(editor(self.data.copy()))
