from typing import Callable
import numpy as np

from vimos.base import Container
from vimos.utils import COCO_I2N


class Skeleton(Container):
    def __init__(self, keypoint=None):
        super().__init__(keypoint)

        if keypoint is not None:
            self.dimension = self.data.shape[1] - 3

    def _prepare_data(self, keypoint):
        keypoint_scores = np.expand_dims(keypoint["keypoint_scores"][0], -1)
        keypoints = keypoint["keypoints"][0]

        weight = np.ones_like(keypoint_scores)
        selected = np.ones_like(keypoint_scores)

        data = np.concatenate(
            [keypoints, keypoint_scores, weight, selected],
            axis=1,
        )
        return data

    def get_vertices(self):
        return self.data[self.data[:, -1].astype(bool), : self.dimension]

    def get_weights(self):
        return self.data[self.data[:, -1].astype(bool), -2]

    def get_keypoint_scores(self):
        return self.data[self.data[:, -1].astype(bool), -3]

    def __repr__(self):
        names = [COCO_I2N[i] for i in sorted(np.where(self.data[:, -1])[0])]
        vertices = self.get_vertices()
        visible = self.get_keypoint_scores()

        string = "Skeleton([\n"
        for i in range(len(names)):
            string += f"\t{names[i].ljust(11)}: {tuple(vertices[i].astype(int))}, scores: {visible[i]:.2f}\n"
        string += "])"
        return string
