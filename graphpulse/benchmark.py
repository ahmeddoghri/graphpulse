from __future__ import annotations

from .data import make_graph
from .score import auc, degree_baseline, graphpulse_score


def main() -> None:
    graph = make_graph()
    pulse = auc(graph.anomaly, graphpulse_score(graph))
    degree = auc(graph.anomaly, degree_baseline(graph))
    print("graphpulse benchmark: heterophily-aware graph anomaly scoring")
    print(f"degree_auc      {degree:.3f}")
    print(f"graphpulse_auc  {pulse:.3f}")
    print(f"auc_gain        {pulse - degree:.3f}")
    print(f"nodes           {len(graph.labels)}")
    print(f"edges           {len(graph.edges)}")


if __name__ == "__main__":
    main()
