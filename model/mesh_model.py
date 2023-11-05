import torch
import numpy as np

from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator

class FrankMocap():
    def __init__(self):
        smpl_dir = './extra_data/'    
        checkpoint_path ='./extra_data/body_module/pretrained_weights/2020_05_31-00_50_43-best-51.749683916568756.pt'

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)
        #assert torch.cuda.is_available(), "Current version only supports GPU"

        self.body_bbox_detector = BodyPoseEstimator()
        self.body_mocap = BodyMocap(checkpoint_path, smpl_dir, device, False)

    def process(self, img, key = None):
        body_pose_list, body_bbox_list = self.body_bbox_detector.detect_body_pose(img)
        hand_bbox_list = [None, ] * len(body_bbox_list)
        
        #Sort the bbox using bbox size 
        # (to make the order as consistent as possible without tracking)
        bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
        idx_big2small = np.argsort(bbox_size)[::-1]
        body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]

        if len(body_bbox_list) == 0:
            return False, None
        
        #body_bbox_list = body_bbox_list[0]   

        # Body Pose Regression
        result = self.body_mocap.regress(img, body_bbox_list)[0]
        
        if key is not None:
            result = result[key]#getattr(result, key)
            
        return True, result