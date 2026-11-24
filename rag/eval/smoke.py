"""Smoke test of retrieval on the seed pilot questions.

    python -m rag.eval.smoke [--variant default] [--k 5]

For each question: is ANY candidate gold id in the top-k? Misses are reported as
"needs label review" with the retrieved ids and scores (the gold ids are provisional and not
lawyer-checked; do not tune chunking or labels to force a pass). Writes reports/smoke_test.md.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from rag.config import ROOT, get_settings
from rag.eval.questions import load_questions


def run_smoke(variant: str | None = None, k: int = 5) -> dict:
    from rag.retrieval.retriever import get_retriever

    s = get_settings(variant)
    qs = load_questions(ROOT / s.raw["eval"]["questions"])
    r = get_retriever(variant)
    rows = []
    for q in qs:
        chunks = r.retrieve(q.question, k=k)
        hit_ranks = [c.rank for c in chunks if q.gold & set(c.unit_ids)]
        rows.append({"question_id": q.question_id, "language": q.language, "gold": sorted(q.gold),
                     "hit": bool(hit_ranks), "first_hit_rank": hit_ranks[0] if hit_ranks else None,
                     "retrieved": [(c.chunk_id, round(c.score, 4)) for c in chunks]})
    lines = [f"# Retrieval smoke test (top-{k})", "",
             f"- run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"- chunking variant: `{s.chunking.name}`, corpus_version: `{r.store.corpus_version}`, "
             f"embedding: `{s.embedding_provider}:{s.embedding_model}`",
             "- gold ids are provisional candidates (not lawyer-checked); a miss means **needs label review**, "
             "not a failure to tune away.", "",
             "| question | language | candidate gold | result | first hit rank |", "|---|---|---|---|---|"]
    for x in rows:
        res = "hit" if x["hit"] else "MISS - needs label review"
        lines.append(f"| {x['question_id']} | {x['language']} | {', '.join(x['gold'])} | {res} | {x['first_hit_rank'] or '-'} |")
    lines.append("")
    for x in rows:
        lines.append(f"## {x['question_id']} ({'hit' if x['hit'] else 'needs label review'})")
        lines.append("")
        for cid, sc in x["retrieved"]:
            lines.append(f"- `{cid}` score {sc}")
        lines.append("")
    hits = sum(x["hit"] for x in rows)
    lines.insert(2, f"**{hits}/{len(rows)} questions have a candidate gold id in the top {k}.**\n")
    out = s.path("reports_dir") / ("smoke_test.md" if s.chunking.name == "default" else f"variants/{s.chunking.name}/smoke_test.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rows": rows, "hits": hits, "n": len(rows), "report": str(out)}


def main(argv=None) -> int:
    for stream in (sys.stdout,):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args(argv)
    res = run_smoke(a.variant, a.k)
    for x in res["rows"]:
        status = "hit " if x["hit"] else "MISS (needs label review)"
        print(f"{x['question_id']:<9} {status:<26} gold={','.join(x['gold'])}")
        if not x["hit"]:
            for cid, sc in x["retrieved"]:
                print(f"           retrieved {cid:<28} {sc}")
    print(f"{res['hits']}/{res['n']} with a candidate gold id in top {a.k}; report: {res['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
