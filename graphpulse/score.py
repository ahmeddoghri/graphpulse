from __future__ import annotations

from .data import Graph


def graphpulse_score(graph: Graph) -> list[float]:
    adj = graph.neighbors()
    scores = []
    for idx, neigh in enumerate(adj):
        if not neigh:
            scores.append(0.0)
            continue
        neighbor_mean = sum(graph.features[j] for j in neigh) / len(neigh)
        residual = abs(graph.features[idx] - neighbor_mean)
        cross = sum(1 for j in neigh if graph.labels[j] != graph.labels[idx]) / len(neigh)
        degree_penalty = 1.0 / (1.0 + len(neigh))
        scores.append(0.72 * residual + 0.25 * cross + 0.03 * degree_penalty)
    return scores


def degree_baseline(graph: Graph) -> list[float]:
    adj = graph.neighbors()
    return [float(len(neigh)) for neigh in adj]


def auc(labels: list[int], scores: list[float]) -> float:
    pos = [(score, idx) for idx, (label, score) in enumerate(zip(labels, scores)) if label == 1]
    neg = [(score, idx) for idx, (label, score) in enumerate(zip(labels, scores)) if label == 0]
    wins = 0.0
    total = len(pos) * len(neg)
    for p, _ in pos:
        for n, _ in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total
