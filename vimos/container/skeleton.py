import numpy as np

from vimos.base import Container


class Skeleton(Container):
    def __init__(self, keypoint=None):
        super().__init__(keypoint)

        if keypoint is not None:
            self.dimension = self.data.shape[1] - 2

    def _prepare_data(self, keypoint):
        keypoint_scores = np.expand_dims(keypoint.keypoint_scores[0], -1)
        keypoints_visible = np.expand_dims(keypoint.keypoints_visible[0], -1)
        keypoints = keypoint.keypoints[0]

        data = np.concatenate([keypoints, keypoint_scores, keypoints_visible], axis=1)
        return data

    def get_keypoints(self):
        return self.data[:, : self.dimension]

    def get_keypoint_scores(self):
        return self.data[:, -2]

    def get_keypoints_visible(self):
        return self.data[:, -1]
