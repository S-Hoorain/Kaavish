"""End-to-end retrieval smoke test on the five seed pilot questions.

Needs the built index (python -m rag.cli ingest && python -m rag.cli index). A question whose
candidate gold ids are not in the top 5 is reported as "needs label review" (warning + report),
NOT a test failure: the gold ids are provisional and must not be tuned to force a pass.
"""
from __future__ import annotations

import warnings

import pytest

from rag.config import get_settings


def _index_ready() -> bool:
    try:
        from rag.index.store import get_collection

        return get_collection(get_settings(), check_model=True).count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _index_ready(), reason="index not built: python -m rag.cli ingest && python -m rag.cli index")
def test_seed_questions_top5():
    from rag.eval.smoke import run_smoke

    res = run_smoke(k=5)
    assert res["n"] == 5
    for row in res["rows"]:
        assert len(row["retrieved"]) == 5, "retrieval must return k results"
        assert all(0.0 <= sc <= 1.0 for _, sc in row["retrieved"])
        if not row["hit"]:
            warnings.warn(f"{row['question_id']}: needs label review; gold={row['gold']} "
                          f"retrieved={row['retrieved']}")
    print(f"\n{res['hits']}/{res['n']} seed questions have a candidate gold id in the top 5 "
          f"(details: {res['report']})")
