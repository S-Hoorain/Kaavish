"""Retrieval metrics against a small example worked out by hand (FR-EVAL-03 acceptance)."""
from __future__ import annotations

import pytest

from rag.eval.metrics import aggregate, hit_at_k, ndcg_at_k, question_metrics, recall_at_k, reciprocal_rank

# ranks:      1        2        3        4 (dup)  5      6 (merged chunk)
RANKED = [["A:1"], ["B:2"], ["G:1"], ["G:1"], ["X:9"], ["G:2", "Y:1"]]
GOLD = {"G:1", "G:2", "G:3"}


def test_hit_and_recall():
    assert hit_at_k(RANKED, GOLD, 2) == 0.0
    assert hit_at_k(RANKED, GOLD, 5) == 1.0          # G:1 at rank 3
    assert recall_at_k(RANKED, GOLD, 5) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, GOLD, 10) == pytest.approx(2 / 3)   # G:2 via merged chunk at rank 6


def test_mrr():
    assert reciprocal_rank(RANKED, GOLD) == pytest.approx(1 / 3)
    assert reciprocal_rank([["A"]], GOLD) == 0.0


def test_ndcg_by_hand():
    # DCG = 1/log2(3+1) + 1/log2(6+1) = 0.5 + 0.356207 = 0.856207 (rank 4 repeats G:1 -> no gain)
    # IDCG (3 gold, k=10) = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.630930 + 0.5 = 2.130930
    assert ndcg_at_k(RANKED, GOLD, 10) == pytest.approx(0.856207 / 2.130930, abs=1e-5)
    assert ndcg_at_k([["G:1"], ["G:2"], ["G:3"]], GOLD, 10) == pytest.approx(1.0)


def test_aggregate_means():
    a = question_metrics([["G:1"]], {"G:1"})
    b = question_metrics([["Z"]], {"G:1"})
    agg = aggregate([a, b])
    assert agg["hit@5"] == 0.5 and agg["mrr"] == 0.5 and agg["recall@10"] == 0.5
