from typing import Union

import cv2
import numpy as np

from vimos.base import Container, Editor


class Photo(Container):
    def apply(self, editor: Editor, inplace: bool = False):
        if inplace:
            self.data = editor(self.data)
            return self
        else:
            return Photo(editor(self.data.copy()))

    def _prepare_data(self, input_source: Union[str, np.ndarray]):
        if isinstance(input_source, str):
            data = cv2.imread(input_source)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        else:
            data = input_source
        return data
