"""Retrieval metrics over parent units (FR-EVAL-03).

Input for one question: `ranked` = the retrieved chunks in rank order, each given as the list of
parent unit ids it covers (a merged chunk covers several, a split piece covers one), and `gold` =
the set of gold parent ids.

* hit@k     1 if ANY gold id is covered by the top-k chunks, else 0 (the task's "hit" definition)
* recall@k  |gold covered by top-k| / |gold|
* MRR       1 / rank of the first chunk covering any gold id (0 if none in the list)
* nDCG@k    binary gain: a chunk gains 1 if it covers at least one gold id not covered by an
            earlier chunk (pieces of an already-found section add nothing);
            IDCG = sum_{i=1..min(|gold|,k)} 1/log2(i+1)
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _covered(ranked: Sequence[Iterable[str]], k: int) -> set[str]:
    out: set[str] = set()
    for units in ranked[:k]:
        out.update(units)
    return out


def hit_at_k(ranked: Sequence[Iterable[str]], gold: set[str], k: int) -> float:
    return 1.0 if gold & _covered(ranked, k) else 0.0


def recall_at_k(ranked: Sequence[Iterable[str]], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & _covered(ranked, k)) / len(gold)


def reciprocal_rank(ranked: Sequence[Iterable[str]], gold: set[str]) -> float:
    for i, units in enumerate(ranked, start=1):
        if gold & set(units):
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[Iterable[str]], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for i, units in enumerate(ranked[:k], start=1):
        new = (gold & set(units)) - seen
        if new:
            dcg += 1.0 / math.log2(i + 1)
            seen |= new
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def question_metrics(ranked: Sequence[Iterable[str]], gold: set[str]) -> dict[str, float]:
    return {
        "hit@5": hit_at_k(ranked, gold, 5),
        "hit@10": hit_at_k(ranked, gold, 10),
        "recall@5": recall_at_k(ranked, gold, 5),
        "recall@10": recall_at_k(ranked, gold, 10),
        "mrr": reciprocal_rank(ranked[:10], gold),
        "ndcg@10": ndcg_at_k(ranked, gold, 10),
    }


METRIC_NAMES = ["hit@5", "hit@10", "recall@5", "recall@10", "mrr", "ndcg@10"]


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {m: float("nan") for m in METRIC_NAMES}
    return {m: sum(r[m] for r in rows) / len(rows) for m in METRIC_NAMES}
