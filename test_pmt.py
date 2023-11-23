import cv2
import numpy as np

from vimos.container import Photo, Album

from vimos.editor import ResizeEditor
from vimos.modifier import SelectModifier, WeightModifier
from vimos.model import build_pose_model
from vimos.task import PoseMatchingTask, TaskConfig
from vimos.utils import Pipeline


def visualize_pmt(frame, reference_container, score, output):
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    image = np.copy(reference_container.data)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (
            image.shape[1] * frame.shape[0] // image.shape[0],
            frame.shape[0],
        ),
    )

    mask = np.zeros_like(image, np.uint8)
    mask = cv2.rectangle(
        mask,
        (0, int(image.shape[0] * (1 - 0.025 / score))),
        (image.shape[1], image.shape[0]),
        (0, 0, 255),
        -1,
    )
    image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    image = cv2.putText(
        image,
        "Matched" if output else "Not Matched", 
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if output else (0, 0, 255),
        2,
    )
    frame = np.concatenate([frame, image], axis=1)
    return frame


if __name__ == "__main__":
    # 1. Container 정의 (레퍼런스 이미지)
    reference_container = Photo("sample/pingpong/pose4.jpg")

    # 2. Editor 정의 (이미지 전처리)
    editor = ResizeEditor(0.5, 0.5)
    
    # 3. Model 정의 (포즈 추출)
    model = build_pose_model("deeppose_resnet", model_complexity=0)

    # 4. Modifier 정의 (포즈 후처리)
    modifier = Pipeline([
        SelectModifier(["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"]), 
        WeightModifier({"l_shoulder": 1, "r_shoulder": 1, "l_elbow": 2, "r_elbow": 2, "l_wrist": 4, "r_wrist": 4})
    ])
    
    # 5. Task 정의 (포즈 유사도 랭킹)
    task_config = TaskConfig(
        model=model,
        editor=editor,
        modifier=modifier,
        metric="euclidean",
    )
    task = PoseMatchingTask(
        task_config,
        reference=reference_container,
        threshold=0.025,
    )
    task.start()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        output, score, _ = task.process(Photo(frame))

        visualized_frame = visualize_pmt(frame, reference_container, score, output)
        cv2.imshow("frame", visualized_frame)

        if cv2.waitKey(1) == 27:
            break
