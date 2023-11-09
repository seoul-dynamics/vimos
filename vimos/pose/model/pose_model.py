import torch
from mmpose.apis import init_model, inference_topdown

from vimos.base import Model
from vimos.container import Photo, Skeleton


class PoseModel(Model):
    def __init__(self, model_cfg, model_ckpt, dimension, device):
        self.model = init_model(model_cfg, model_ckpt, device=device)
        self.device = device
        self.dimension = dimension

    def process(self, input_image: Photo):
        output = inference_topdown(self.model, input_image.data)
        output = [Skeleton(output[i].pred_instances) for i in range(len(output))]

        if len(output) == 1:
            output = output[0]

        return output
