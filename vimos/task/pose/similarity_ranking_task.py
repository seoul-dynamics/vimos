import multiprocessing as mp

from vimos.base import Task
from vimos.container import Photo, Album
from vimos.metric import build_metric


class SimilarityRankingTask(Task):
    def __init__(self, task_config, reference, k=None, similar_front=True):
        self.task_config = task_config
        self.reference = reference
        self.k = k
        self.similar_front = similar_front

    def start(self):
        self.queue_in, self.queue_out = mp.Queue(), mp.Queue()

        model = self.task_config.model
        model.connect(self.queue_in, self.queue_out)

        mp.Process(target=model.run).start()

        if type(self.reference) is Photo:
            self.reference = Album(self.reference)

        self.queue_in.put(self.reference)
        self.reference = self.queue_out.get()

        if type(self.task_config.metric) is str:
            self.task_config.metric = build_metric(self.task_config.metric)

    def process(self, input_data):
        self.queue_in.put(input_data)
        skeleton = self.queue_out.get()

        scores = [self.task_config.metric(skeleton, ref) for ref in self.reference]
        output = zip(scores, range(len(self.reference)))

        reverse = self.similar_front ^ self.task_config.metric.similar_small
        output = sorted(output, key=lambda x: x[0], reverse=reverse)[: self.k]

        scores, indexes = zip(*output)
        return indexes, scores
