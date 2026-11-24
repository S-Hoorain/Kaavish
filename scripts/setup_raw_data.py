"""Populate data/raw/ with copies of the inputs that live in Code/ (which is read-only).

Run once from the project root:

    python scripts/setup_raw_data.py

What it does
* Copies each source file listed in MAPPING from Code/ to data/raw/ (copies, not symlinks:
  symlinks need admin rights on Windows). Copied files are marked read-only.
* Never overwrites: if a destination exists with the same SHA-256 it is skipped; if it exists
  with a different SHA-256 the script stops and reports it (data/raw must not be modified).
* Writes data/raw/RAW_MANIFEST.json (source path -> data/raw path, sha256, bytes).
* Creates data/raw/source_dates.yaml from a template if it does not exist yet.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "Code"
RAW = ROOT / "data" / "raw"

ACTS = {
    # doc_id: (folder under Code/data, pdf file name in Code/)
    "MFLO-1961": ("MFLO-1961", "MFLO, 1961.pdf"),
    "FCA-1964": ("FCA-1964", "The family courts act, 1964.pdf"),
    "DMMA-1939": ("DMMA-1939", "DMMA, 1939.pdf"),
    "GWA-1890": ("GWA-1890", "The guardians and wards act, 1890.pdf"),
}


def build_mapping() -> list[tuple[Path, Path]]:
    m: list[tuple[Path, Path]] = []
    m.append((CODE / "data" / "articles.jsonl", RAW / "constitution" / "articles.jsonl"))
    m.append((CODE / "constitution.pdf", RAW / "constitution" / "constitution.pdf"))
    m.append((CODE / "data" / "constitution" / "meta.json", RAW / "constitution" / "meta.json"))
    for doc_id, (folder, pdf) in ACTS.items():
        m.append((CODE / "data" / folder / "sections.jsonl", RAW / "acts" / f"{doc_id}.jsonl"))
        m.append((CODE / "data" / folder / "meta.json", RAW / "acts" / "meta" / f"{doc_id}.meta.json"))
        m.append((CODE / pdf, RAW / "acts" / "pdf" / f"{doc_id}.pdf"))
    for cls, folder, prefix in (("long", "longform_judgements", "judgement_"),
                                ("short", "shortform_judgements", "short_judgement_")):
        for pdf in sorted((CODE / folder).glob(f"{prefix}*.pdf")):
            m.append((pdf, RAW / "judgments" / cls / pdf.name))
            js = pdf.with_suffix(".json")
            if js.exists():
                m.append((js, RAW / "judgments" / cls / "existing_json" / js.name))
        rep = CODE / folder / "json_output" / "report.md"
        if rep.exists():
            m.append((rep, RAW / "judgments" / cls / "existing_json" / "report.md"))
    return m


SOURCE_DATES_TEMPLATE = """\
# "As of" date of each statute source: the date up to which the copy we ingested is current.
# Fill every TODO from the source document or its official page. Do not guess.
# Format: free text exactly as printed on the source (e.g. "10-3-2022"), plus where it was read.
# The ingestion pipeline copies `as_of` into every chunk of that document.
MFLO-1961:
  as_of: "10-3-2022"
  basis: "PDF page header: 'Updated till 10-3-2022'"
FCA-1964:
  as_of: "17.1.2025"
  basis: "PDF page header: 'FINAL COPY Updated till 17.1.2025'"
DMMA-1939:
  as_of: "29-05-2024"
  basis: "Document date 29-05-2024 (not an 'updated till' date; confirm)"
GWA-1890:
  as_of: TODO
  basis: "No 'updated till' date found in the PDF. Needs a human to confirm from the official source."
CONST-1973:
  as_of: TODO
  basis: "PDF says 'As amended upto the Twenty-Seventh Amendment' (no date). Needs a human to confirm."
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    if not CODE.exists():
        print(f"ERROR: {CODE} not found. Run from the project root next to Code/.")
        return 2
    manifest = []
    problems = []
    for src, dst in build_mapping():
        if not src.exists():
            problems.append(f"missing source: {src.relative_to(ROOT)}")
            continue
        s_hash = sha256(src)
        if dst.exists():
            d_hash = sha256(dst)
            if d_hash != s_hash:
                problems.append(f"DIFFERS (not overwritten): {dst.relative_to(ROOT)} vs {src.relative_to(ROOT)}")
            action = "exists"
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            dst.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            action = "copied"
        manifest.append({
            "source": src.relative_to(ROOT).as_posix(),
            "raw_path": dst.relative_to(ROOT).as_posix(),
            "sha256": s_hash,
            "bytes": src.stat().st_size,
            "action": action,
        })
    RAW.mkdir(parents=True, exist_ok=True)
    man_path = RAW / "RAW_MANIFEST.json"
    if man_path.exists():
        man_path.chmod(stat.S_IWRITE | stat.S_IREAD)
    man_path.write_text(json.dumps({"files": manifest}, indent=2), encoding="utf-8")
    sd = RAW / "source_dates.yaml"
    if not sd.exists():
        sd.write_text(SOURCE_DATES_TEMPLATE, encoding="utf-8")
        print("created data/raw/source_dates.yaml (template; fill the TODOs)")
    copied = sum(1 for m in manifest if m["action"] == "copied")
    print(f"{len(manifest)} files mapped ({copied} copied, {len(manifest) - copied} already present)")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
