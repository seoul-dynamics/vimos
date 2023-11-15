import cv2

from vimos.container import Photo, Album

# from vimos.editor import HumanCropEditor
# from vimos.modifier import SelectVertexModifier, WeightEdgeModifier
from vimos.model import build_pose_model
from vimos.task import SimilarityRankingTask, PoseMatchingTask, TaskConfig
from vimos.utils import Pipeline


if __name__ == "__main__":
    reference = Album(["sample/pose1.png", "sample/pose2.png", "sample/pose3.png"])
    """
    editor = HumanCropEditor(256, 256)
    modifier = Pipeline(
        [SelectVertexModifier([1, 3, 5]), WeightEdgeModifier([0.5, 0.5])]
    )
    """

    task_config = TaskConfig(
        model=build_pose_model("deeppose_resnet", model_complexity=0),
        metric="euclidean",
        # editor=editor,
        # modifier=modifier,
    )
    task = SimilarityRankingTask(
        task_config,
        reference=reference,
        k=3,
        similar_front=True,
    )

    reference = Photo("sample/pose1.png")
    task = PoseMatchingTask(task_config, reference=reference, threshold=0.5)

    task.start()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # frame = Photo(frame)
        frame = Photo("sample/pose1.png")

        output = task.process(frame)
        print(output)

        if cv2.waitKey(1) == 27:
            break
