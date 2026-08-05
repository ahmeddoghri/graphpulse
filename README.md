# graphpulse

The most dangerous node in your graph has a perfectly average number of connections. Degree-based detectors will never catch it. graphpulse does.

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

Most anomaly detectors reach for degree first because it is the easiest
number on a graph to compute, which also makes it the easiest number to
sneak past. graphpulse plants exactly that failure mode: nodes that look
ordinary by degree while every one of their neighbors belongs to the wrong
community. The detector scores the residual between a node's own features
and its neighborhood average, adds a small cross-community signal, and stays
simple enough that you can read the whole scoring function on your phone.

## Run it

```bash
git clone https://github.com/ahmeddoghri/graphpulse
cd graphpulse
pip install -e ".[dev]"
python -m graphpulse.benchmark
```

## Verified benchmark

Generated locally with `python -m graphpulse.benchmark`:

```text
degree_auc      0.196
graphpulse_auc  0.938
auc_gain        0.741
nodes           96
edges           394
```

Degree scores 0.196 AUC on this graph, which is worse than a coin flip and
proof that popularity is not the same as suspicion. graphpulse scores 0.938,
a 0.741 point gain on a graph of 96 nodes and 394 edges. This is not trying
to out-leaderboard a GNN, it is the sharp, honest baseline you check before
you spend a training budget on one.

**Update:** 0.938 leans hard on a signal a real detector would never have.
The dominant term reads each node's ground-truth community label
directly, the exact assignment used to plant which nodes are anomalous.
Isolated, that one term alone gets a perfect AUC 1.0, a readout of the
construction rule, not a graph signal. Replace it with a community
estimate inferred from each node's own observable feature (never the
oracle label) and mean AUC drops from ~0.94 to ~0.66, still well ahead of
degree, but a fraction of the published number. `python -m
graphpulse.eval_v2` runs the honest comparison. Details below.

## The score reads the answer key it's supposed to be inferring

`graphpulse_score`'s dominant term computes the fraction of a node's
neighbors whose `graph.labels` differ from its own. `labels` is the
ground-truth community assignment `make_graph` uses to *plant* the
anomalies: every anomalous node is built with 5 of its 6 edges going to
the opposite community, on purpose. Checking `graph.labels` directly
isn't a graph signal, it's the exact rule used to construct the label,
read back out.

```python
from graphpulse.data import make_graph
from graphpulse.score import auc

g = make_graph()

def cross_only(graph):
    adj = graph.neighbors()
    return [
        sum(1 for j in n if graph.labels[j] != graph.labels[i]) / len(n) if n else 0.0
        for i, n in enumerate(adj)
    ]

auc(g.anomaly, cross_only(g))   # 1.0
```

One term, isolated, perfect separation. In any real deployment you would
not know a node's true community before you'd already solved anomaly
detection; that's the thing the detector exists to help you figure out.

```bash
python -m graphpulse.eval_v2
```
```
seeds        n    degree   oracle (v1)   fair (v2)
tuning      15     0.203         0.932       0.674
holdout     15     0.211         0.934       0.658
```

`graphpulse/score_v2.py` replaces the oracle lookup with a community
estimate inferred from each node's own observable feature value
(thresholded at the midpoint), the only thing genuinely available at
scoring time. Mean AUC drops from ~0.94 to ~0.66, confirmed on a disjoint
15-seed holdout evaluated exactly once, still comfortably ahead of the
degree baseline (~0.20) and better than chance, but a fraction of what
the published number implies about how easy this problem is without the
oracle. `data.py`/`score.py` are untouched, and the published table
above still reproduces exactly; `graphpulse_score_v2` is opt-in.

## Research trail

- Deep graph anomaly detection survey, 2024: https://arxiv.org/abs/2409.09957
- Learning from graphs with heterophily survey, 2024: https://arxiv.org/abs/2401.09769
- GOODAT test-time graph OOD detection, 2024: https://arxiv.org/html/2401.06176v1
- UB-GOLD graph anomaly and OOD benchmark, 2024: https://arxiv.org/html/2406.15523v1

## Tests

```bash
pytest -q
ruff check .
```

MIT © Ahmed Doghri
