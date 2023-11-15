import numpy as np
from scipy.spatial import procrustes

from vimos.base import Metric


class EuclideanMetric(Metric):
    similar_small = True
    min_score = 0
    max_score = 1

    def __call__(self, ref_skeleton, query_skeleton):
        ref_vertices = ref_skeleton.get_vertices()
        query_vertices = query_skeleton.get_vertices()

        ref_vertices, query_vertices = self._normalize(ref_vertices, query_vertices)
        weights = ref_skeleton.get_weights()

        total_score = 0
        for ref_vertex, query_vertex, weight in zip(
            ref_vertices, query_vertices, weights
        ):
            score = self._evaluate(ref_vertex, query_vertex)
            total_score += score * weight

        total_score = total_score / len(ref_vertices)
        total_score = (total_score - self.min_score) / (self.max_score - self.min_score)
        return total_score

    def _evaluate(self, ref_vectex, query_vertex):
        return np.linalg.norm(ref_vectex - query_vertex)

    def _normalize(self, ref_vertices, query_vertices):
        ref_vertices = np.array(ref_vertices)
        query_vertices = np.array(query_vertices)

        ref_vertices, query_vertices, _ = procrustes(ref_vertices, query_vertices)
        return ref_vertices, query_vertices
