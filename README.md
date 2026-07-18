# graphpulse

The most dangerous node in your graph has a perfectly average number of connections. Degree-based detectors will never catch it. graphpulse does.

![CI](https://github.com/ahmeddoghri/graphpulse/actions/workflows/ci.yml/badge.svg)
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
