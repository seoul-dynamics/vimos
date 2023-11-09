import numpy as np

from vimos.base import Container


class Skeleton(Container):
    def _prepare_data(self, keypoint):
        keypoint_scores = np.expand_dims(keypoint.keypoint_scores[0], -1)
        keypoints_visible = np.expand_dims(keypoint.keypoints_visible[0], -1)
        keypoints = keypoint.keypoints[0]

        data = np.concatenate([keypoints, keypoint_scores, keypoints_visible], axis=1)
        return data
