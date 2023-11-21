from typing import Union

import torch
import numpy as np

from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator

from vimos.base import Model
from vimos.container import Photo, Album, Skeleton

class FrankMocap():
    def __init__(self, device):
        self.device = device
        
        self.body_bbox_detector = BodyPoseEstimator()
        self.body_mocap = BodyMocap(device)

    def process(self, input_image: Union[Photo, Album], verbose=False):
        img_list = (
            [input_image.data]
            if type(input_image) is Photo
            else [image.data for image in input_image.data]
        )
        output = []

        for img in img_list:
            _, body_bbox_list = self.body_bbox_detector.detect_body_pose(img)
            
            if len(body_bbox_list) == 0:
                return False, None
            
            result = self.body_mocap.regress(img, body_bbox_list)
            if not verbose:
                result = [person_result['pred_vertices_img'] for person_result in result]
                output.append(result)
            
            else:
                output.append(result)
            
        if len(img_list) == 1:
            return True, output[0]
        
        return True, output