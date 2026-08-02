"""The cross-community signal reads the answer key. This doesn't.

``graphpulse_score``'s dominant term, ``cross``, computes the fraction of
a node's neighbors whose ``graph.labels`` differ from its own. ``labels``
is the ground-truth community assignment ``make_graph`` used to *plant*
the anomalies in the first place: every anomalous node is constructed
with 5 of its 6 edges going to the opposite community. Reading
``graph.labels`` directly is not a graph signal, it's an oracle. Isolated
on its own, that one term alone gets **AUC 1.0**, because it's a direct
readout of the exact rule used to build the label.

In any real deployment you would not know a node's true community before
you've done the anomaly detection; that's the thing you're trying to
figure out. ``community_from_features`` infers it instead, from each
node's own observable feature value (thresholded at the midpoint), which
is the only thing genuinely available at scoring time. ``graphpulse_score_v2``
is the same weighted combination as the original, with the oracle
``graph.labels`` lookup replaced by that inferred community.

Isolated, the inferred-community cross term alone: AUC ~0.73, down from
the oracle's 1.0. The full combined score: AUC ~0.66 on average across
seeds, down from ~0.94, still comfortably ahead of the degree baseline
(~0.20) and better than chance, but a fraction of what the published
number implies about how easy this problem actually is without the
oracle.
"""
from __future__ import annotations

from .data import Graph


def community_from_features(graph: Graph, threshold: float = 0.5) -> list[int]:
    """Infer each node's community from its own observable feature value,
    never from ``graph.labels``. The only information a real detector
    would have at scoring time."""
    return [1 if f > threshold else 0 for f in graph.features]


def graphpulse_score_v2(graph: Graph) -> list[float]:
    adj = graph.neighbors()
    inferred = community_from_features(graph)
    scores = []
    for idx, neigh in enumerate(adj):
        if not neigh:
            scores.append(0.0)
            continue
        neighbor_mean = sum(graph.features[j] for j in neigh) / len(neigh)
        residual = abs(graph.features[idx] - neighbor_mean)
        cross = sum(1 for j in neigh if inferred[j] != inferred[idx]) / len(neigh)
        degree_penalty = 1.0 / (1.0 + len(neigh))
        scores.append(0.72 * residual + 0.25 * cross + 0.03 * degree_penalty)
    return scores
