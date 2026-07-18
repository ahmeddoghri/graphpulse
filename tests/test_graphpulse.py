from graphpulse import auc, degree_baseline, graphpulse_score, make_graph


def test_graph_has_planted_anomalies() -> None:
    graph = make_graph()
    assert sum(graph.anomaly) == 12
    assert len(graph.neighbors()) == len(graph.labels)


def test_auc_bounds() -> None:
    graph = make_graph()
    value = auc(graph.anomaly, graphpulse_score(graph))
    assert 0.0 <= value <= 1.0


def test_graphpulse_beats_degree_baseline() -> None:
    graph = make_graph()
    assert auc(graph.anomaly, graphpulse_score(graph)) > auc(graph.anomaly, degree_baseline(graph))
