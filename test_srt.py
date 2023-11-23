import cv2
import numpy as np

from vimos.container import Photo, Album

from vimos.editor import ResizeEditor
from vimos.modifier import SelectModifier, WeightModifier
from vimos.model import build_pose_model
from vimos.task import SimilarityRankingTask, TaskConfig
from vimos.utils import Pipeline


def visualize_srt(frame, reference_container, score):
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    temp = []
    for i in range(4):
        image = np.copy(reference_container[i].data)
        image = image[: 512, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (
                frame.shape[1] // 4,
                image.shape[0] * frame.shape[1] // 4 // image.shape[1],
            ),
        )
        image = cv2.putText(
            image,
            f"{score[i] * 100:.0f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        mask = np.zeros_like(image, np.uint8)
        mask = cv2.rectangle(
            mask,
            (0, int(image.shape[0] * (1 - score[i]))),
            (image.shape[1], image.shape[0]),
            (0, 0, 255),
            -1,
        )
        image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        temp.append(image)

    temp = np.concatenate(temp, axis=1)
    frame = np.concatenate([frame, temp], axis=0)

    return frame


if __name__ == "__main__":
    # 1. Container 정의 (레퍼런스 이미지)
    reference_container = Album(
        [
            "sample/weight_training/pose1.jpg",
            "sample/weight_training/pose2.jpg",
            "sample/weight_training/pose3.jpg",
            "sample/weight_training/pose4.jpg",
        ]
    )

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
    task = SimilarityRankingTask(
        task_config,
        reference=reference_container,
        k=4,
        similar_front=True,
        beta=5, 
    )
    task.start()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        output, score, _ = task.process(Photo(frame))

        visualized_frame = visualize_srt(frame, reference_container, score)
        cv2.imshow("frame", visualized_frame)

        if cv2.waitKey(1) == 27:
            break
