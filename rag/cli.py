"""Command line interface.

    python -m rag.cli ingest   [--variant default]
    python -m rag.cli index    [--variant default]
    python -m rag.cli retrieve "question" [--k 5]
    python -m rag.cli ask      "question" --mode c1 --k 5 [--json]
    python -m rag.cli chunk    "GWA-1890:29#1"
"""
from __future__ import annotations

import argparse
import json
import sys


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def cmd_ingest(a) -> int:
    from rag.ingest.run import run_ingestion

    r = run_ingestion(a.variant)
    print(f"report: reports/{'' if (a.variant or 'default') == 'default' else 'variants/' + a.variant + '/'}ingestion_report.md")
    return 0 if r["chunks"] else 1


def cmd_index(a) -> int:
    from rag.index.build import build_index

    build_index(a.variant)
    return 0


def cmd_retrieve(a) -> int:
    from rag.retrieval.retriever import get_retriever

    rr = get_retriever(a.variant).retrieve_with_trace(a.question, k=a.k)
    print(f"stages: {rr.stages}  timings_ms: {rr.timings_ms}")
    for c in rr.chunks:
        print(f"{c.rank:>2}. {c.score:.4f}  {c.chunk_id:<28} {c.text[:110].replace(chr(10), ' ')}")
    return 0


def cmd_ask(a) -> int:
    from rag.pipeline import ask

    try:
        r = ask(a.question, mode=a.mode, k=a.k, variant=a.variant)
    except RuntimeError as e:  # already written to logs/requests.jsonl with retrieval and timings
        print(f"ERROR: {e}\n(the request, including retrieval results, was logged to logs/requests.jsonl)")
        return 2
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0 if r.get("status") == "ok" else 1
    print(f"request_id: {r['request_id']}  mode: {r['mode']}  corpus: {r.get('corpus_version')}  "
          f"prompt: {r['prompt_version']} ({r['prompt_sha256']})")
    if r["mode"] == "c1":
        print("\nretrieved:")
        for x in r["retrieved"]:
            print(f"  {x['rank']:>2}. {x['score']:.4f}  {x['chunk_id']}")
    if r.get("status") != "ok":
        print(f"\nERROR: {r.get('error')}")
        for att in r.get("llm_attempts", []):
            print(f"  attempt {att['attempt']}: {att['parse']}")
        return 1
    f = r["final"]
    print(f"\nanswer ({f['language']}, insufficient_evidence={f['insufficient_evidence']}):\n{f['answer']}\n")
    print("claims:")
    for i, c in enumerate(f["claims"], start=1):
        mark = {True: "VERIFIED", False: "FAILED", None: "n/a"}[c["verified"]]
        print(f"  [{i}] {mark:<8} {c.get('chunk_id')}  ({c['reason']})\n      claim: {c['text']}")
        if c.get("quote"):
            print(f"      quote: \"{c['quote']}\"")
    v = r["verification"]
    print(f"\nverification: {v['n_verified']}/{v['n_checked']} quotes verified "
          f"(rate={v['verification_rate']}); claims={v['n_claims']}")
    print(f"latency_ms: {r['latency_ms']}  tokens: {r['tokens']}  cost: {r['cost']}")
    print("logged to logs/requests.jsonl")
    return 0


def cmd_chunk(a) -> int:
    from rag.config import get_settings
    from rag.index.store import load_chunk_store

    store = load_chunk_store(str(get_settings(a.variant).chunks_path))
    c = store.get(a.chunk_id)
    if not c:
        print(f"unknown chunk_id {a.chunk_id!r}")
        return 1
    print(json.dumps(c, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    p = argparse.ArgumentParser(prog="python -m rag.cli", description="Law or Fiqh? basic RAG baseline (R0 / C1) and C0 control")
    p.add_argument("--variant", default=None, help="chunking variant (config/chunking_variants.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="parse inputs and write data/processed/<variant>/chunks.jsonl + report")
    sub.add_parser("index", help="embed chunks and build the Chroma collection")
    r = sub.add_parser("retrieve", help="dense retrieval only")
    r.add_argument("question")
    r.add_argument("--k", type=int, default=5)
    q = sub.add_parser("ask", help="answer a question in C0 or C1")
    q.add_argument("question")
    q.add_argument("--mode", choices=["c0", "c1"], default="c1")
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--json", action="store_true", help="print the full JSON result")
    c = sub.add_parser("chunk", help="print one chunk record")
    c.add_argument("chunk_id")
    for sp in sub.choices.values():
        sp.add_argument("--variant", default=argparse.SUPPRESS, help="chunking variant")
    a = p.parse_args(argv)
    return {"ingest": cmd_ingest, "index": cmd_index, "retrieve": cmd_retrieve, "ask": cmd_ask,
            "chunk": cmd_chunk}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
