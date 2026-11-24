"""reports/ingestion_report.json + reports/ingestion_report.md (quality gate for ingestion)."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rag.config import ROOT, Settings

BINS = [0, 60, 120, 250, 400, 512, 600, 10_000]


def _bin(n: int) -> str:
    for lo, hi in zip(BINS, BINS[1:]):
        if lo <= n < hi:
            return f"{lo}-{hi - 1}" if hi < 10_000 else f">={lo}"
    return "?"


def report_dir(settings: Settings) -> Path:
    base = settings.path("reports_dir")
    return base if settings.chunking.name == "default" else base / "variants" / settings.chunking.name


def write_ingestion_report(*, settings: Settings, chunks: list[dict], skipped: list[dict], rejected: list[dict],
                           statute_stats: dict, judgments: list[dict], judgment_errors: list[dict],
                           judgment_excluded: list[dict], corpus_version: str, source_dates: dict,
                           elapsed: float) -> dict[str, Any]:
    cfg = settings.chunking
    per_doc: dict[str, dict] = defaultdict(lambda: Counter())
    for c in chunks:
        d = per_doc[c["doc_id"]]
        d["chunks"] += 1
        d[f"unit_type:{c['unit_type']}"] += 1
        d["tokens"] += c["token_count"]
    hist_all = Counter(_bin(c["token_count"]) for c in chunks)
    hist_by_type = {t: dict(Counter(_bin(c["token_count"]) for c in chunks if c["doc_type"] == t))
                    for t in sorted({c["doc_type"] for c in chunks})}
    over_cap = [{"chunk_id": c["chunk_id"], "token_count": c["token_count"]} for c in chunks
                if c["token_count"] > cfg.hard_cap_tokens]
    trunc = [{"chunk_id": c["chunk_id"], "embed_token_count": c["embed_token_count"]} for c in chunks
             if c["embed_token_count"] + 2 > settings.raw["embeddings"]["max_seq_length"]]
    as_of_todo = sorted({c["doc_id"] for c in chunks if "as_of_todo" in (c.get("flags") or [])})

    j_summary = []
    for j in judgments:
        h = j["header"]
        j_summary.append({
            "doc_id": j["doc_id"], "file": j["source_file"], "length_class": j["length_class"],
            "pages": j["n_pages"], "paragraphs": len(j["paragraphs"]),
            "numbering": Counter(p["numbering"] for p in j["paragraphs"]).most_common(1)[0][0] if j["paragraphs"] else None,
            "footnotes": len(j["footnotes"]), "title": j["title"], "body_start": j["body_start_method"],
            "case_numbers": h.get("case_numbers"), "date": h.get("date"), "judge": h.get("judge"),
            "parties": f'{h.get("petitioner")} v. {h.get("respondent")}',
            "unparsed_header_fields": h.get("unparsed"), "header_conflicts_with_existing_json": h.get("conflicts"),
            "field_sources": h.get("_sources"), "anomalies": j["anomalies"],
            "gaps_vs_existing_json": j["gaps_vs_existing_json"], "manifest_notes": j.get("manifest_notes"),
            "scan_check": j["scan_check"],
        })

    rng = random.Random(20261124)
    samples = {}
    by_type = defaultdict(list)
    for c in chunks:
        by_type[c["doc_type"]].append(c)
    for t, cs in sorted(by_type.items()):
        pick = rng.sample(cs, min(10, len(cs)))
        samples[t] = [{k: c[k] for k in ("chunk_id", "parent_id", "unit_type", "token_count", "embed_token_count",
                                          "contains_quote", "cites_cases", "cites_statutes", "embed_text", "text")}
                      for c in pick]

    report = {
        "corpus_version": corpus_version,
        "chunking_variant": cfg.name,
        "chunking": cfg.as_dict(),
        "elapsed_seconds": round(elapsed, 1),
        "totals": {"chunks": len(chunks), "skipped_records": len(skipped), "rejected_chunks": len(rejected),
                   "judgments_parsed": len(judgments), "judgment_errors": len(judgment_errors),
                   "judgments_excluded": len(judgment_excluded)},
        "per_document": {k: dict(v) for k, v in sorted(per_doc.items())},
        "statute_records": statute_stats,
        "token_histogram_all": dict(sorted(hist_all.items(), key=lambda kv: int(kv[0].split("-")[0].lstrip(">=")))),
        "token_histogram_by_doc_type": hist_by_type,
        "chunks_over_cap": over_cap,
        "chunks_truncated_at_embedding": trunc,
        "as_of_todo_documents": as_of_todo,
        "source_dates": source_dates,
        "rejected": rejected,
        "skipped": [{k: v for k, v in s_.items() if k != "text"} for s_ in skipped],
        "judgments": j_summary,
        "judgment_errors": judgment_errors,
        "judgments_excluded": judgment_excluded,
        "samples": samples,
    }
    out = report_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ingestion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "ingestion_report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(r: dict) -> str:
    L: list[str] = []
    a = L.append
    t = r["totals"]
    a(f"# Ingestion report\n")
    a(f"- corpus_version: `{r['corpus_version']}`  \n- chunking variant: `{r['chunking_variant']}` "
      f"({json.dumps(r['chunking'])})  \n- time: {r['elapsed_seconds']} s\n")
    a("## Totals\n")
    a("| chunks | skipped records | rejected chunks | judgments parsed | judgment errors | judgments excluded |")
    a("|---|---|---|---|---|---|")
    a(f"| {t['chunks']} | {t['skipped_records']} | {t['rejected_chunks']} | {t['judgments_parsed']} | "
      f"{t['judgment_errors']} | {t['judgments_excluded']} |\n")
    a("## Statute records per document\n")
    a("| doc_id | records | skipped (status) | units indexed | units split | chunks |")
    a("|---|---|---|---|---|---|")
    for d, s in r["statute_records"].items():
        a(f"| {d} | {s['records']} | {s['skipped']} | {s['units_indexed']} | {s['units_split']} | {s['chunks']} |")
    a("\n## Chunks per document\n")
    a("| doc_id | chunks | unit types | mean tokens |")
    a("|---|---|---|---|")
    for d, s in r["per_document"].items():
        types = ", ".join(f"{k.split(':')[1]}={v}" for k, v in s.items() if k.startswith("unit_type:"))
        a(f"| {d} | {s['chunks']} | {types} | {s['tokens'] // max(1, s['chunks'])} |")
    a("\n## Token length histogram (BGE-M3 tokens of `text`)\n")
    a("| bin | all | " + " | ".join(r["token_histogram_by_doc_type"]) + " |")
    a("|---|---|" + "---|" * len(r["token_histogram_by_doc_type"]))
    for b, n in r["token_histogram_all"].items():
        a(f"| {b} | {n} | " + " | ".join(str(v.get(b, 0)) for v in r["token_histogram_by_doc_type"].values()) + " |")
    a(f"\n## Chunks over the hard cap ({r['chunking']['hard_cap_tokens']} tokens): {len(r['chunks_over_cap'])}\n")
    for c in r["chunks_over_cap"]:
        a(f"- `{c['chunk_id']}`: {c['token_count']} tokens")
    a(f"\n## Chunks whose embed_text is truncated by the 512-token embedding window: {len(r['chunks_truncated_at_embedding'])}\n")
    for c in r["chunks_truncated_at_embedding"]:
        a(f"- `{c['chunk_id']}`: {c['embed_token_count']} tokens (+2 special)")
    a(f"\n## Missing-metadata rejects: {len(r['rejected'])}\n")
    for c in r["rejected"]:
        a(f"- `{c['chunk_id']}` missing {c['missing']}")
    if r["as_of_todo_documents"]:
        a(f"\n**as_of still TODO in data/raw/source_dates.yaml (indexed, flagged):** {', '.join(r['as_of_todo_documents'])}\n")
    a(f"\n## Skipped statute records: {len(r['skipped'])}\n")
    a("| unit | status | title |")
    a("|---|---|---|")
    for s in r["skipped"]:
        a(f"| {s['unit_id']} | {s['status']} | {(s.get('title') or '').replace('|', '/')} |")
    a("\n## Judgments\n")
    a("| doc_id | file | pages | paras | numbering | footnotes | date | case | unparsed header fields | conflicts with existing JSON |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for j in r["judgments"]:
        conf = "; ".join(f"{c['field']}: json={c['existing_json']!r} pdf={c['pdf_parse']!r}" for c in j["header_conflicts_with_existing_json"] or [])
        a(f"| {j['doc_id']} | {Path(j['file']).name} | {j['pages']} | {j['paragraphs']} | {j['numbering']} | "
          f"{j['footnotes']} | {j['date']} | {', '.join(j['case_numbers'] or [])} | {', '.join(j['unparsed_header_fields'] or []) or '-'} | {conf.replace('|', '/') or '-'} |")
    a("\n### Parser anomalies and gaps in the existing JSONs (per judgment)\n")
    for j in r["judgments"]:
        a(f"**{j['doc_id']}** ({Path(j['file']).name}){' - ' + j['manifest_notes'] if j.get('manifest_notes') else ''}")
        for x in j["anomalies"]:
            a(f"- anomaly: {x}")
        for x in j["gaps_vs_existing_json"]:
            a(f"- gap: {x}")
        a("")
    if r["judgment_errors"]:
        a("\n### Judgment errors\n")
        for e in r["judgment_errors"]:
            a(f"- {e['file']}: {e['error']}")
    a("\n## Random chunks for human inspection (10 per document type, seed 20261124)\n")
    for t, cs in r["samples"].items():
        a(f"\n### {t}\n")
        for c in cs:
            a(f"#### `{c['chunk_id']}` ({c['unit_type']}, {c['token_count']} tokens, contains_quote={c['contains_quote']})\n")
            if c["cites_cases"] or c["cites_statutes"]:
                a(f"cites_cases: {c['cites_cases']}  \ncites_statutes: {c['cites_statutes']}\n")
            a(f"embed header: `{c['embed_text'].splitlines()[0] if c['embed_text'] else ''}`\n")
            a("text (verbatim):\n```text\n" + c["text"] + "\n```\n")
    return "\n".join(L) + "\n"
