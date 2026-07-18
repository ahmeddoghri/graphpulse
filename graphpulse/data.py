from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Graph:
    labels: list[int]
    features: list[float]
    edges: list[tuple[int, int]]
    anomaly: list[int]

    def neighbors(self) -> list[list[int]]:
        adj = [[] for _ in self.labels]
        for a, b in self.edges:
            adj[a].append(b)
            adj[b].append(a)
        return adj


def make_graph(seed: int = 8, normal: int = 84, anomalies: int = 12) -> Graph:
    rng = random.Random(seed)
    labels = [0 if idx < normal // 2 else 1 for idx in range(normal)]
    features = [label + rng.gauss(0.0, 0.08) for label in labels]
    anomaly = [0] * normal
    edges: list[tuple[int, int]] = []
    for i in range(normal):
        for j in range(i + 1, normal):
            same = labels[i] == labels[j]
            if rng.random() < (0.16 if same else 0.018):
                edges.append((i, j))
    for idx in range(anomalies):
        node = normal + idx
        planted_label = idx % 2
        labels.append(planted_label)
        features.append(1.0 - planted_label + rng.gauss(0.0, 0.05))
        anomaly.append(1)
        opposite = [i for i, label in enumerate(labels[:normal]) if label != planted_label]
        same = [i for i, label in enumerate(labels[:normal]) if label == planted_label]
        for target in rng.sample(opposite, 5):
            edges.append((node, target))
        for target in rng.sample(same, 1):
            edges.append((node, target))
    return Graph(labels, features, edges, anomaly)
