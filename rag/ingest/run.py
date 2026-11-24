"""Ingestion entry point: statutes + Constitution + judgments -> validated chunks.jsonl + reports.

    python -m rag.cli ingest [--variant default]
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from rag.config import ROOT, Settings, get_settings
from rag.ingest.judgments.build import PARSER_VERSION, judgment_chunks, parse_judgment
from rag.ingest.judgments.manifest import doc_id_for, read_manifest, write_default_manifest
from rag.ingest.judgments.pdf_extract import ScannedPDFError
from rag.ingest.report import write_ingestion_report
from rag.ingest.statutes import ingest_statutes
from rag.schema import missing_fields, order_record
from rag.tokens import bge_m3_counter

CHUNKER_VERSION = "chunker-v1"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_ingestion(variant: str | None = None, verbose: bool = True) -> dict[str, Any]:
    t0 = time.time()
    s: Settings = get_settings(variant)
    cfg = s.chunking
    raw = s.path("raw_dir")
    docs_cfg = yaml.safe_load((ROOT / "config" / "documents.yaml").read_text(encoding="utf-8"))
    source_dates = yaml.safe_load(s.path("source_dates").read_text(encoding="utf-8")) or {}
    count = bge_m3_counter(s.raw["embeddings"]["bge_m3_model"])
    log = print if verbose else (lambda *a, **k: None)

    # ---- statutes -------------------------------------------------------------------------
    log("[ingest] statutes and Constitution ...")
    st_chunks, skipped, st_stats = ingest_statutes(raw, docs_cfg["statutes"], source_dates,
                                                   s.raw["corpus"]["skip_statuses"], count, cfg)

    # ---- judgments -------------------------------------------------------------------------
    man_path = s.path("judgment_manifest")
    if not man_path.exists():
        write_default_manifest(raw / "judgments", man_path)
        log(f"[ingest] wrote default {man_path.relative_to(ROOT)} (include=y for all; review the notes)")
    manifest = read_manifest(man_path)
    jcfg = docs_cfg["judgments"]
    parsed_dir = s.path("processed_dir") / "judgments_parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    j_chunks: list[dict] = []
    j_parsed: list[dict] = []
    j_errors: list[dict] = []
    j_excluded: list[dict] = []
    input_files: list[Path] = [man_path, ROOT / "config" / "documents.yaml", s.path("source_dates")]
    for d in docs_cfg["statutes"].values():
        input_files.append(raw / d["file"])
    for row in manifest:
        if row["include"].lower() not in ("y", "yes"):
            j_excluded.append({"file": row["file"], "reason": f"include={row['include']!r} in manifest", "notes": row["notes"]})
            continue
        pdf = raw / "judgments" / row["file"]
        ex = pdf.parent / "existing_json" / (pdf.stem + ".json")
        doc_id = doc_id_for(row, jcfg["doc_id_prefix"])
        input_files.append(pdf)
        if ex.exists():
            input_files.append(ex)
        try:
            parsed = parse_judgment(pdf, ex if ex.exists() else None, doc_id, row["length_class"])
        except ScannedPDFError as e:
            j_errors.append({"file": row["file"], "doc_id": doc_id, "error": str(e), "action": "STOPPED: scanned PDF"})
            log(f"[ingest] STOP {row['file']}: {e}")
            continue
        parsed["source_file"] = (pdf.relative_to(ROOT)).as_posix()
        parsed["manifest_notes"] = row["notes"]
        (parsed_dir / f"{doc_id}.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=1), encoding="utf-8")
        j_parsed.append(parsed)
        j_chunks.extend(judgment_chunks(parsed, count, cfg, jcfg["court_display"], jcfg["jurisdiction"]))
    log(f"[ingest] judgments parsed: {len(j_parsed)}, errors: {len(j_errors)}, excluded: {len(j_excluded)}")

    # ---- validation ----------------------------------------------------------------------------
    strict = bool(s.raw["corpus"].get("strict_as_of"))
    accepted, rejected = [], []
    for c in st_chunks + j_chunks:
        miss = missing_fields(c, strict_as_of=strict)
        if miss:
            rejected.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "missing": miss,
                             "text_preview": c["text"][:200]})
        else:
            accepted.append(c)

    # ---- corpus version ------------------------------------------------------------------------
    hashes = sorted({(p.relative_to(ROOT).as_posix(), sha256_file(p)) for p in input_files})
    version_src = json.dumps({"inputs": hashes, "chunking": cfg.as_dict(), "chunker": CHUNKER_VERSION,
                              "judgment_parser": PARSER_VERSION,
                              "skip_statuses": s.raw["corpus"]["skip_statuses"], "strict_as_of": strict},
                             sort_keys=True)
    corpus_version = hashlib.sha256(version_src.encode()).hexdigest()[:12]
    for c in accepted:
        c["corpus_version"] = corpus_version
        c["token_count"] = count(c["text"])
        c["embed_token_count"] = count(c["embed_text"])
    accepted = [order_record(c) for c in accepted]

    out = s.processed_dir
    _write_jsonl(out / "chunks.jsonl", accepted)
    _write_jsonl(out / "skipped.jsonl", skipped)
    _write_jsonl(out / "rejected.jsonl", rejected)
    manifest_out = {
        "corpus_version": corpus_version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunking_variant": cfg.name,
        "chunking": cfg.as_dict(),
        "chunker_version": CHUNKER_VERSION,
        "judgment_parser_version": PARSER_VERSION,
        "tokenizer": s.raw["embeddings"]["bge_m3_model"],
        "inputs": [{"path": p, "sha256": h} for p, h in hashes],
        "counts": {"chunks": len(accepted), "skipped_records": len(skipped), "rejected_chunks": len(rejected),
                   "judgments_parsed": len(j_parsed), "judgment_errors": len(j_errors)},
    }
    (out / "corpus_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")

    report = write_ingestion_report(
        settings=s, chunks=accepted, skipped=skipped, rejected=rejected, statute_stats=st_stats,
        judgments=j_parsed, judgment_errors=j_errors, judgment_excluded=j_excluded,
        corpus_version=corpus_version, source_dates=source_dates, elapsed=time.time() - t0)
    log(f"[ingest] {len(accepted)} chunks (corpus_version {corpus_version}); "
        f"{len(skipped)} skipped records; {len(rejected)} rejected chunks -> {out.relative_to(ROOT)}")
    return {"corpus_version": corpus_version, "chunks": len(accepted), "report": report}
