class Pipeline:
    def __init__(self, callables):
        self.callables = callables

    def __call__(self, data):
        for callable in self.callables:
            data = callable(data)
        return data


COCO_I2N = {
    0: "nose",
    1: "l_eye",
    2: "r_eye",
    3: "l_ear",
    4: "r_ear",
    5: "l_shoulder",
    6: "r_shoulder",
    7: "l_elbow",
    8: "r_elbow",
    9: "l_wrist",
    10: "r_wrist",
    11: "l_hip",
    12: "r_hip",
    13: "l_knee",
    14: "r_knee",
    15: "l_ankle",
    16: "r_ankle",
}

COCO_N2I = {v: k for k, v in COCO_I2N.items()}
