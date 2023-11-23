from typing import Union

from mmpose.apis import init_model, inference_topdown, inference_pose_lifter_model

from vimos.base import Model
from vimos.container import Photo, Album, Skeleton


class PoseModel(Model):
    def __init__(self, model_cfg, model_ckpt, dimension, device):
        self.model = init_model(model_cfg, model_ckpt, device=device)
        self.dimension = dimension

        if dimension == "3d":
            sub_model_cfg = "vimos/configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_8xb64-210e_coco-256x192.py"
            sub_model_ckpt = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_regression/coco/td-reg_res50_8xb64-210e_coco-256x192-72ef04f3_20220913.pth"
            
            self.sub_model = init_model(sub_model_cfg, sub_model_ckpt, device=device)


    def predict(self, input_image: Union[Photo, Album]):
        input_images = (
            [input_image.data]
            if type(input_image) is Photo
            else [image.data for image in input_image.data]
        )
        if self.dimension == "2d":
            output = [inference_topdown(self.model, image)[0].pred_instances for image in input_images]
        else:
            output = [inference_topdown(self.sub_model, image) for image in input_images]
            output = [
                inference_pose_lifter_model(
                    self.model, 
                    [skeleton], 
                    with_track_id=False,
                )
                for skeleton in output
            ]

        output = [Skeleton(output[i][0]) for i in range(len(output))]

        if len(output) == 1:
            output = output[0]

        return output
