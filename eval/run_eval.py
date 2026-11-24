"""Evaluation harness: retrieval metrics + C0/C1 answers for manual grading.

    python eval/run_eval.py                                  # default variant, 3 runs, C0 and C1
    python eval/run_eval.py --retrieval-only                 # no LLM calls
    python eval/run_eval.py --chunking-variant no_header     # compare chunking variants
    python eval/run_eval.py --runs 1 --modes c1 --k 5

Outputs (default variant; other variants go to reports/variants/<name>/ for answers):
    reports/retrieval_metrics.csv        Recall@5/@10, MRR, nDCG@10 (+ hit@k) per language and overall,
                                         one block of rows per chunking variant (rows of other variants kept)
    reports/retrieval_per_question.csv   per question: retrieved ids/scores and per-question metrics
    reports/answers_c0_c1.csv            every C0/C1 answer with verification rate, latency, tokens, cost,
                                         and empty grading columns
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.config import get_settings  # noqa: E402
from rag.eval.metrics import METRIC_NAMES, aggregate, question_metrics  # noqa: E402
from rag.eval.questions import load_questions  # noqa: E402

RETRIEVAL_COLUMNS = ["chunking_variant", "language", "n_questions", *METRIC_NAMES, "corpus_version",
                     "embedding_model", "k_retrieved", "eval_run_id", "timestamp_utc"]
PERQ_COLUMNS = ["chunking_variant", "question_id", "language", "question_type", "gold", "hit@5", "hit@10",
                "recall@5", "recall@10", "mrr", "ndcg@10", "first_gold_rank", "retrieved", "label_status",
                "eval_run_id"]
ANSWER_COLUMNS = ["eval_run_id", "chunking_variant", "question_id", "language", "question_type", "mode", "run",
                  "request_id", "status", "error", "answer", "answer_language", "insufficient_evidence",
                  "n_claims", "n_checked", "n_verified", "verification_rate", "failed_claims", "claims_json",
                  "retrieved_chunk_ids", "gold_hit_in_context", "latency_embed_ms", "latency_search_ms",
                  "latency_generate_ms", "latency_verify_ms", "latency_total_ms", "input_tokens", "output_tokens",
                  "cost_usd", "cost_note", "llm_provider", "llm_base_url_host", "llm_model", "llm_reasoning_effort", "prompt_version", "corpus_version",
                  "grader", "grade_correctness_1to5", "grade_citations_1to5", "grade_notes"]


def _rewrite_keeping_other_variants(path: Path, columns: list[str], rows: list[dict], variant: str) -> None:
    kept = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            kept = [r for r in csv.DictReader(f) if r.get("chunking_variant") != variant]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept + rows)


def ensure_index(variant: str, rebuild_chunks: bool) -> None:
    from rag.index.build import build_index
    from rag.ingest.run import run_ingestion

    s = get_settings(variant)
    if rebuild_chunks or not s.chunks_path.exists():
        run_ingestion(variant)
    build_index(variant)  # no-op when the collection already matches corpus_version and model


def run_retrieval(questions, variant: str, k_retrieved: int, run_id: str) -> tuple[list[dict], list[dict]]:
    from rag.retrieval.retriever import get_retriever

    s = get_settings(variant)
    r = get_retriever(variant)
    perq, by_lang = [], {}
    for q in questions:
        chunks = r.retrieve(q.question, k=k_retrieved)
        ranked = [c.unit_ids for c in chunks]
        m = question_metrics(ranked, q.gold)
        first = next((c.rank for c in chunks if q.gold & set(c.unit_ids)), None)
        perq.append({"chunking_variant": variant, "question_id": q.question_id, "language": q.language,
                     "question_type": q.question_type, "gold": ";".join(sorted(q.gold)),
                     **{k: round(v, 4) for k, v in m.items()}, "first_gold_rank": first,
                     "retrieved": ";".join(f"{c.chunk_id}={c.score:.4f}" for c in chunks),
                     "label_status": q.label_status, "eval_run_id": run_id})
        by_lang.setdefault(q.language, []).append(m)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {"chunking_variant": variant, "corpus_version": r.store.corpus_version,
              "embedding_model": f"{s.embedding_provider}:{s.embedding_model}", "k_retrieved": k_retrieved,
              "eval_run_id": run_id, "timestamp_utc": now}
    agg_rows = []
    for lang, rows in sorted(by_lang.items()):
        agg_rows.append({"language": lang, "n_questions": len(rows),
                         **{k: round(v, 4) for k, v in aggregate(rows).items()}, **common})
    all_rows = [m for rows in by_lang.values() for m in rows]
    agg_rows.append({"language": "overall", "n_questions": len(all_rows),
                     **{k: round(v, 4) for k, v in aggregate(all_rows).items()}, **common})
    return agg_rows, perq


def run_answers(questions, variant: str, modes: list[str], runs: int, k: int, run_id: str,
                request_delay_seconds: float = 0) -> list[dict]:
    from rag.pipeline import ask

    s = get_settings(variant)
    rows = []
    total = len(questions) * len(modes) * runs
    n = 0
    for q in questions:
        for mode in modes:
            for run in range(1, runs + 1):
                n += 1
                print(f"[answers] {n}/{total} {q.question_id} {mode} run {run}", flush=True)
                if request_delay_seconds > 0:
                    time.sleep(request_delay_seconds)
                base = {"eval_run_id": run_id, "chunking_variant": variant, "question_id": q.question_id,
                        "language": q.language, "question_type": q.question_type, "mode": mode, "run": run,
                        "llm_provider": s.llm_provider, "llm_base_url_host": s.llm_base_url_host,
                        "llm_model": s.llm_model, "llm_reasoning_effort": s.llm_reasoning_effort or None}
                try:
                    r = ask(q.question, mode=mode, k=k, variant=variant,
                            extra_log={"eval_run_id": run_id, "question_id": q.question_id, "run": run})
                except Exception as e:  # already logged by ask(); keep the row so nothing is silently dropped
                    rows.append({**base, "status": "error", "error": str(e)})
                    continue
                f = r.get("final") or {}
                v = r.get("verification") or {}
                lat = r.get("latency_ms") or {}
                ids = [x["chunk_id"] for x in r.get("retrieved") or []]
                units = {u for x in r.get("retrieved") or [] for u in x.get("unit_ids") or []}
                rows.append({**base, "request_id": r["request_id"], "status": r.get("status"), "error": r.get("error"),
                             "answer": f.get("answer"), "answer_language": f.get("language"),
                             "insufficient_evidence": f.get("insufficient_evidence"),
                             "n_claims": v.get("n_claims"), "n_checked": v.get("n_checked"),
                             "n_verified": v.get("n_verified"), "verification_rate": v.get("verification_rate"),
                             "failed_claims": json.dumps(v.get("failures") or [], ensure_ascii=False),
                             "claims_json": json.dumps(f.get("claims") or [], ensure_ascii=False),
                             "retrieved_chunk_ids": ";".join(ids),
                             "gold_hit_in_context": (bool(q.gold & units) if mode == "c1" else None),
                             "latency_embed_ms": lat.get("embed"), "latency_search_ms": lat.get("search"),
                             "latency_generate_ms": lat.get("generate"), "latency_verify_ms": lat.get("verify"),
                             "latency_total_ms": lat.get("total"),
                             "input_tokens": (r.get("tokens") or {}).get("input"),
                             "output_tokens": (r.get("tokens") or {}).get("output"),
                             "cost_usd": (r.get("cost") or {}).get("cost_usd"),
                             "cost_note": (r.get("cost") or {}).get("cost_note"),
                             "prompt_version": r.get("prompt_version"), "corpus_version": r.get("corpus_version")})
    return rows


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    s0 = get_settings()
    ev = s0.raw["eval"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default=ev["questions"])
    ap.add_argument("--chunking-variant", default=s0.raw["chunking"]["variant"])
    ap.add_argument("--runs", type=int, default=ev["runs"])
    ap.add_argument("--modes", default=",".join(ev["modes"]))
    ap.add_argument("--k", type=int, default=s0.raw["retrieval"]["default_k"], help="passages given to the LLM in C1")
    ap.add_argument("--k-retrieved", type=int, default=ev["retrieval_k"], help="depth for retrieval metrics (>=10)")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--rebuild-chunks", action="store_true", help="re-run ingestion for the variant first")
    ap.add_argument("--request-delay-seconds", type=float, default=0,
                    help="pause before each LLM answer call (useful for provider RPM limits)")
    a = ap.parse_args(argv)

    variant = a.chunking_variant
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    questions = load_questions(ROOT / a.questions)
    print(f"[eval] run {run_id}: {len(questions)} questions, variant={variant}")
    t0 = time.time()
    ensure_index(variant, a.rebuild_chunks)

    agg, perq = run_retrieval(questions, variant, max(10, a.k_retrieved), run_id)
    reports = get_settings(variant).path("reports_dir")
    _rewrite_keeping_other_variants(reports / "retrieval_metrics.csv", RETRIEVAL_COLUMNS, agg, variant)
    _rewrite_keeping_other_variants(reports / "retrieval_per_question.csv", PERQ_COLUMNS, perq, variant)
    print("[eval] retrieval metrics:")
    for r in agg:
        print(f"   {r['language']:<11} n={r['n_questions']}  " + "  ".join(f"{m}={r[m]:.3f}" for m in METRIC_NAMES))

    if a.retrieval_only:
        print(f"[eval] retrieval only; done in {time.time() - t0:.0f}s")
        return 0
    s = get_settings(variant)
    if not (s.llm_provider and s.llm_model):
        print("[eval] LLM not configured (LLM_PROVIDER / LLM_MODEL in .env): skipping the C0/C1 answer runs. "
              "Run: python scripts/check_env.py")
        return 3
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    rows = run_answers(questions, variant, modes, a.runs, a.k, run_id, a.request_delay_seconds)
    out = reports if variant == "default" else reports / "variants" / variant
    out.mkdir(parents=True, exist_ok=True)
    path = out / "answers_c0_c1.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ANSWER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for mode in modes:
        mr = [r for r in rows if r["mode"] == mode and r.get("status") == "ok"]
        rates = [r["verification_rate"] for r in mr if r.get("verification_rate") is not None]
        lat = [r["latency_total_ms"] for r in mr if r.get("latency_total_ms") is not None]
        print(f"[eval] {mode}: {len(mr)}/{sum(1 for r in rows if r['mode'] == mode)} ok; "
              f"mean verification rate={sum(rates) / len(rates) if rates else 'n/a'}; "
              f"mean latency ms={sum(lat) / len(lat) if lat else 'n/a'}")
    print(f"[eval] wrote {path.relative_to(ROOT)} in {time.time() - t0:.0f}s")
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
