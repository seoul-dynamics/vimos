from typing import Union

from mmpose.apis import init_model, inference_topdown

from vimos.base import Model
from vimos.container import Photo, Album, Skeleton


class PoseModel(Model):
    def __init__(self, model_cfg, model_ckpt, dimension, device):
        self.model = init_model(model_cfg, model_ckpt, device=device)
        self.device = device
        self.dimension = dimension

    def process(self, input_image: Union[Photo, Album]):
        input_images = (
            [input_image.data]
            if type(input_image) is Photo
            else [image.data for image in input_image.data]
        )

        output = [inference_topdown(self.model, image) for image in input_images]
        output = [Skeleton(output[i][0].pred_instances) for i in range(len(output))]

        if len(output) == 1:
            output = output[0]

        return output
