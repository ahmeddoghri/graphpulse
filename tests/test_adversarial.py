"""Tests for the oracle-label finding and the fair, feature-inferred fix."""

from __future__ import annotations

from graphpulse.adversarial import HOLDOUT_SEEDS, TUNING_SEEDS
from graphpulse.data import make_graph
from graphpulse.eval_v2 import _summarize, build_report
from graphpulse.score import auc, degree_baseline, graphpulse_score
from graphpulse.score_v2 import community_from_features, graphpulse_score_v2

# --- the finding: the dominant signal is an oracle label read --------------

def test_cross_term_alone_reads_the_planted_construction_rule():
    """Isolate the cross-community term (graph.labels[j] != graph.labels[idx])
    with no other signal: it alone achieves perfect separation, because
    it's a direct readout of how anomalies were constructed (5 of 6 edges
    to the opposite community), not a graph signal."""
    g = make_graph()

    def cross_only(graph):
        adj = graph.neighbors()
        out = []
        for idx, neigh in enumerate(adj):
            if not neigh:
                out.append(0.0)
                continue
            cross = sum(1 for j in neigh if graph.labels[j] != graph.labels[idx]) / len(neigh)
            out.append(cross)
        return out

    assert auc(g.anomaly, cross_only(g)) == 1.0


def test_features_are_a_noisy_proxy_for_the_oracle_label():
    """Confirms why community_from_features works at all: features are
    label + small noise, not an independent observable, so thresholding
    them recovers most (not all) of the community signal."""
    g = make_graph()
    inferred = community_from_features(g)
    agreement = sum(1 for a, b in zip(inferred, g.labels) if a == b) / len(g.labels)
    assert 0.75 < agreement < 1.0  # noisy, not exact


# --- the fix: infer community instead of reading the oracle ----------------

def test_fair_score_drops_substantially_from_the_oracle_score():
    g = make_graph()
    oracle_auc = auc(g.anomaly, graphpulse_score(g))
    fair_auc = auc(g.anomaly, graphpulse_score_v2(g))
    assert fair_auc < oracle_auc - 0.15


def test_fair_score_still_clearly_beats_the_degree_baseline():
    g = make_graph()
    fair_auc = auc(g.anomaly, graphpulse_score_v2(g))
    degree_auc = auc(g.anomaly, degree_baseline(g))
    assert fair_auc > degree_auc + 0.3


def test_fair_score_is_stable_and_above_chance_across_seeds():
    result = _summarize(TUNING_SEEDS)
    assert result["mean_fair_auc"] > 0.6
    assert result["mean_fair_auc"] < result["mean_oracle_auc"] - 0.2


# --- held out, evaluated once ------------------------------------------------

def test_holdout_seeds_are_disjoint_from_tuning_seeds():
    assert not (set(TUNING_SEEDS) & set(HOLDOUT_SEEDS))


def test_holdout_confirms_the_pattern():
    result = _summarize(HOLDOUT_SEEDS)
    assert result["mean_fair_auc"] > 0.6
    assert result["mean_fair_auc"] < result["mean_oracle_auc"] - 0.2


# --- the original module is untouched ---------------------------------------

def test_original_score_module_untouched():
    import graphpulse.score as score_module

    assert not hasattr(score_module, "graphpulse_score_v2")


def test_original_benchmark_still_reproduces():
    g = make_graph()
    assert round(auc(g.anomaly, degree_baseline(g)), 3) == 0.196
    assert round(auc(g.anomaly, graphpulse_score(g)), 3) == 0.938


# --- the full report ---------------------------------------------------------

def test_report_is_reproducible():
    assert build_report() == build_report()
