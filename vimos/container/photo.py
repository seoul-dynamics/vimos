from typing import Union, Callable

import cv2
import numpy as np

from vimos.base import Container, Editor


class Photo(Container):
    def _prepare_data(self, input_source: Union[str, np.ndarray]):
        if isinstance(input_source, str):
            data = cv2.imread(input_source)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        else:
            data = input_source
        return data
