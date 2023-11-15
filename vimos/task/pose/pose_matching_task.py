import multiprocessing as mp

from vimos.base import Task
from vimos.container import Photo, Album
from vimos.metric import build_metric


class PoseMatchingTask(Task):
    def __init__(self, task_config, reference, threshold=0.5):
        self.task_config = task_config
        self.reference = reference
        self.threshold = threshold

    def start(self):
        self.queue_in, self.queue_out = mp.Queue(), mp.Queue()

        model = self.task_config.model
        model.connect(self.queue_in, self.queue_out)

        mp.Process(target=model.run).start()

        self.queue_in.put(self.reference)
        self.reference = self.queue_out.get()

        if type(self.task_config.metric) is str:
            self.task_config.metric = build_metric(self.task_config.metric)

    def process(self, input_data):
        self.queue_in.put(input_data)
        skeleton = self.queue_out.get()

        score = self.task_config.metric(skeleton, self.reference)
        if not self.task_config.metric.similar_small:
            score = 1 - score

        return score < self.threshold, score
