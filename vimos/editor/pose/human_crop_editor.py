import os.path as osp
from mmdet.apis import init_detector, inference_detector

from vimos.base import Editor


class HumanCropEditor(Editor):
    def __init__(self):
        model_cfg = osp.join("vimos", "configs/detection/rtmdet_tiny_8xb32_300e_coco.py")
        model_ckpt = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth"

        self.model = init_detector(model_cfg, model_ckpt, device="cpu")

    def __call__(self, image):
        return image.apply(self._process)
    
    def _process(self, image):
        data = image.data
        data = inference_detector(self.model, data)
        image.data = data
        return image