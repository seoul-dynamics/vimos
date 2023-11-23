import multiprocessing as mp
from copy import deepcopy

from vimos.base import Task
from vimos.container import Photo, Album
from vimos.metric import build_metric


class PoseMatchingTask(Task):
    def __init__(self, task_config, reference, threshold=0.5):
        super().__init__(task_config)
        self.reference = reference
        self.threshold = threshold

    def start(self, num_process=1):
        self.queue_in, self.queue_out = mp.Queue(), mp.Queue()

        model = self.task_config.model
        for _ in range(num_process):
            _model = deepcopy(model)
            _model.connect(self.queue_in, self.queue_out)
            mp.Process(target=_model.run).start()

        if self.task_config.editor is not None:
            self.reference = self.task_config.editor(self.reference)

        self.queue_in.put(self.reference)
        self.reference = self.queue_out.get()

        if self.task_config.modifier is not None:
            self.reference = self.task_config.modifier(self.reference)

        if type(self.task_config.metric) is str:
            self.task_config.metric = build_metric(self.task_config.metric)

    def process(self, input_data):
        skeleton, updated = self._apply(input_data)

        score = self.task_config.metric(skeleton, self.reference)
        if not self.task_config.metric.similar_small:
            score = 1 - score

        return score < self.threshold, score, updated   

    