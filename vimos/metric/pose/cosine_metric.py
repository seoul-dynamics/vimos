import numpy as np

from vimos.base import Metric


class CosineMetric(Metric):
    similar_small = False

    def __call__(self, ref_skeleton, query_skeleton):
        ref_edges = ref_skeleton.get_edges()
        query_edges = query_skeleton.get_edges()

        weights = self.ref_skeleton.get_weights()

        total_score = 0
        for ref_edge, query_edge, weight in zip(ref_edges, query_edges, weights):
            score = self._evaluate(ref_edge, query_edge)
            total_score += score * weight

        total_score = total_score / len(ref_edges)
        return total_score

    def _evaluate(self, ref_edge, query_edge):
        ref_edge = self._normalize(ref_edge)
        query_edge = self._normalize(query_edge)
        return np.dot(ref_edge, query_edge)

    def _normalize(self, edge):
        norm = np.linalg.norm(edge)
        if norm == 0:
            return edge
        return edge / norm
