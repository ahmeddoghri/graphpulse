# graphpulse

A graph anomaly detector for heterophilous failures. Degree baselines miss
nodes that have ordinary-looking degree but the wrong neighborhood. graphpulse
scores the residual between a node feature and its neighborhood, then adds a
small cross-community signal.

![CI](https://github.com/ahmeddoghri/graphpulse/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

## Run it

```bash
git clone https://github.com/ahmeddoghri/graphpulse
cd graphpulse
pip install -e ".[dev]"
python -m graphpulse.benchmark
```

## Verified benchmark

These numbers were generated locally with `python -m graphpulse.benchmark`:

```text
degree_auc      0.196
graphpulse_auc  0.938
auc_gain        0.741
nodes           96
edges           394
```

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
