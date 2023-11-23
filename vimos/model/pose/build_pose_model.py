import os.path as osp

from vimos.model.pose import PoseModel


POSE_MODEL = {
    "deeppose_resnet": {
        "dimension": "2d",
        "max_complexity": 2,
        "model_cfg": [
            "configs/body_2d_keypoint/topdown_regression/coco/td-reg_res50_8xb64-210e_coco-256x192.py",
            "configs/body_2d_keypoint/topdown_regression/coco/td-reg_res101_8xb64-210e_coco-256x192.py",
            "configs/body_2d_keypoint/topdown_regression/coco/td-reg_res152_8xb64-210e_coco-256x192.py",
        ],
        "model_ckpt": [
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_regression/coco/td-reg_res50_8xb64-210e_coco-256x192-72ef04f3_20220913.pth",
            "https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res101_coco_256x192-2f247111_20210205.pth",
            "https://download.openmmlab.com/mmpose/top_down/deeppose/deeppose_res152_coco_256x192-7df89a88_20210205.pth",
        ],
    },
    "yoloxpose": {
        "dimension": "2d",
        "max_complexity": 3,
        "model_cfg": [
            "configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_tiny_4xb64-300e_coco-416.py",
            "configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_s_8xb32-300e_coco-640.py",
            "configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_m_8xb32-300e_coco-640.py",
            "configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_l_8xb32-300e_coco-640.py",
        ],
        "model_ckpt": [
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_tiny_4xb64-300e_coco-416-76eb44ca_20230829.pth",
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_s_8xb32-300e_coco-640-56c79c1f_20230829.pth",
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_m_8xb32-300e_coco-640-84e9a538_20230829.pth",
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_l_8xb32-300e_coco-640-de0f8dee_20230829.pth",
        ],
    },
    "motion_bert": {
        "dimension": "3d",
        "max_complexity": 0,
        "model_cfg": [
            "configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-243frm_8xb32-240e_h36m.py",
        ],
        "model_ckpt": [
            "https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth"
        ],
    },
}


def build_pose_model(model_name, model_complexity=0, device="cpu"):
    assert model_name in POSE_MODEL.keys(), f"Model {model_name} not found"
    assert (
        POSE_MODEL[model_name]["max_complexity"] >= model_complexity
    ), f"Model complexity {model_complexity} not found"

    model_info = POSE_MODEL[model_name]
    model_cfg, model_ckpt = (
        model_info["model_cfg"][model_complexity],
        model_info["model_ckpt"][model_complexity],
    )

    model_cfg = osp.join("vimos", model_cfg)

    return PoseModel(
        model_cfg, model_ckpt, dimension=model_info["dimension"], device=device
    )
