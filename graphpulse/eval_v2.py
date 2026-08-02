"""How much of the 0.938 AUC is the scoring logic, and how much is an
oracle reading the ground-truth label used to plant the anomalies?

``graphpulse.benchmark`` runs once, on the default seed, and reports
``graphpulse_score`` at 0.938 AUC. That score's dominant term reads
``graph.labels`` directly, the exact community assignment used to
construct which nodes are anomalous. This module reruns the same
comparison with that oracle read replaced by ``score_v2``'s
feature-inferred community, across many seeds, to report what the method
achieves without the answer key.

    python -m graphpulse.eval_v2
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Sequence

from .adversarial import HOLDOUT_SEEDS, TUNING_SEEDS
from .data import make_graph
from .score import auc, degree_baseline, graphpulse_score
from .score_v2 import graphpulse_score_v2


def _summarize(seeds: Sequence[int]) -> Dict:
    degree_aucs: List[float] = []
    oracle_aucs: List[float] = []
    fair_aucs: List[float] = []
    for seed in seeds:
        g = make_graph(seed=seed)
        degree_aucs.append(auc(g.anomaly, degree_baseline(g)))
        oracle_aucs.append(auc(g.anomaly, graphpulse_score(g)))
        fair_aucs.append(auc(g.anomaly, graphpulse_score_v2(g)))
    n = len(seeds)
    return {
        "n": n,
        "mean_degree_auc": round(sum(degree_aucs) / n, 4),
        "mean_oracle_auc": round(sum(oracle_aucs) / n, 4),
        "mean_fair_auc": round(sum(fair_aucs) / n, 4),
    }


def build_report() -> Dict:
    return {
        "tuning": _summarize(TUNING_SEEDS),
        "holdout": _summarize(HOLDOUT_SEEDS),
    }


def format_report(report: Dict) -> str:
    lines = [
        "how much of the AUC is the scoring logic vs. an oracle label read?",
        "=" * 68,
        f"{'seeds':<10}{'n':>4}{'degree':>10}{'oracle (v1)':>14}{'fair (v2)':>12}",
        "-" * 68,
    ]
    for name, key in [("tuning", "tuning"), ("holdout", "holdout")]:
        row = report[key]
        lines.append(
            f"{name:<10}{row['n']:>4}{row['mean_degree_auc']:>10.3f}"
            f"{row['mean_oracle_auc']:>14.3f}{row['mean_fair_auc']:>12.3f}"
        )
    lines.append("")
    lines.append(
        "graphpulse_score's dominant term reads graph.labels directly, the exact"
    )
    lines.append(
        "community assignment used to plant the anomalies. isolated, that term"
    )
    lines.append(
        "alone scores AUC 1.0 on its own, a readout of the construction rule, not"
    )
    lines.append(
        "a graph signal. score_v2 infers community from each node's own"
    )
    lines.append(
        "observable feature instead, the only thing a real detector would have."
    )
    lines.append(
        "mean AUC drops from ~0.94 to ~0.66, still well ahead of the degree"
    )
    lines.append(
        "baseline (~0.20) and better than chance, but a fraction of what the"
    )
    lines.append("published number implies about how easy this problem really is.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    report = build_report()
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
