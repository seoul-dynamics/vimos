import multiprocessing as mp
from copy import deepcopy

import numpy as np

from vimos.base import Task
from vimos.container import Photo, Album
from vimos.metric import build_metric


class SimilarityRankingTask(Task):
    def __init__(
        self,
        task_config,
        reference,
        k=None,
        similar_front=True,
        beta=1,
        sort=False,
    ):
        super().__init__(task_config)
        self.reference = reference
        self.k = k
        self.similar_front = similar_front
        self.beta = beta
        self.sort = sort

    def start(self, num_process=1):
        manager = mp.Manager()
        self.queue_in, self.queue_out = manager.Queue(), manager.Queue()
        model = self.task_config.model
        for _ in range(num_process):
            _model = deepcopy(model)
            _model.connect(self.queue_in, self.queue_out)
            mp.Process(target=_model.run).start()

        if type(self.reference) is Photo:
            self.reference = Album(self.reference)

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

        scores = -np.array([self.task_config.metric(skeleton, ref) for ref in self.reference])
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        scores = np.exp(scores * self.beta) / np.sum(np.exp(scores * self.beta), axis=0)
        output = zip(scores, range(len(self.reference)))

        if self.sort:
            reverse = self.similar_front ^ self.task_config.metric.similar_small
            output = sorted(output, key=lambda x: x[0], reverse=reverse)[: self.k]

        scores, indexes = zip(*output)
        return indexes, scores, updated
