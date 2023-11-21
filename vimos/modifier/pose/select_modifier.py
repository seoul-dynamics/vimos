import numpy as np
from mmdet.apis import init_detector, inference_detector

from vimos.base import Modifier
from vimos.utils import COCO_N2I


class SelectModifier(Modifier):
    def __init__(self, landmarks):
        self.landmarks = landmarks
        if isinstance(landmarks[0], str):
            self.landmarks = [COCO_N2I[landmark] for landmark in landmarks]

    def _process(self, skeleton):
        mask = np.zeros_like(skeleton.data[:, -1])
        mask[self.landmarks] = 1

        skeleton.data[:, -1] = mask
        return skeleton
