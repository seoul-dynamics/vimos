from vimos.metric import EuclideanMetric


def build_metric(metric):
    if metric == "euclidean":
        return EuclideanMetric()
    else:
        raise ValueError(f"Unknown metric: {metric}")
