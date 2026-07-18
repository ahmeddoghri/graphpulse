from graphpulse import auc, graphpulse_score, make_graph

graph = make_graph(seed=3)
print(f"auc={auc(graph.anomaly, graphpulse_score(graph)):.3f}")
