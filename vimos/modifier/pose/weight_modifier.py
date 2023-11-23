import numpy as np

from vimos.base import Modifier
from vimos.utils import COCO_N2I


class WeightModifier(Modifier):
    def __init__(self, weights):
        self.weights = weights

    def _process(self, skeleton):
        weight = np.zeros_like(skeleton.data[:, -1])
        for landmark, value in self.weights.items():
            weight[COCO_N2I[landmark]] = value

        weight = weight / np.sum(weight)
        skeleton.data[:, -2] = weight
        return skeleton
