"""data/judgment_manifest.csv: which judgment PDFs are parsed (include=y) and why."""
from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = ["file", "length_class", "include", "notes"]

# Pre-filled notes from reading every judgment (see HANDOFF.md section 9). include stays "y" for all:
# the inclusion decision belongs to the team.
DEFAULT_NOTES = {
    "long/judgement_13.pdf": "REVIEW: civil property dispute (specific performance, s.12(2) CPC), not a family-law matter",
    "short/short_judgement_02.pdf": "REVIEW: Art.199 petition for recovery of a minor alleged to be in illegal detention (habeas-type); family-adjacent",
    "short/short_judgement_03.pdf": "REVIEW: Art.199 petition for production of an adult woman detained by her husband (habeas-type); family-adjacent",
    "short/short_judgement_14.pdf": "REVIEW: Art.199 petition for police recovery of a minor (habeas-type); family-adjacent",
    "short/short_judgement_15.pdf": "REVIEW: Art.199 petition for recovery of minors, settled by compromise (habeas-type); family-adjacent",
    "short/short_judgement_13.pdf": "Family suit counter-claim over third-party property (Family Courts Act Schedule); in scope but property-focused",
}


def write_default_manifest(raw_judgments: Path, path: Path) -> None:
    rows = []
    for cls in ("long", "short"):
        for pdf in sorted((raw_judgments / cls).glob("*.pdf")):
            rel = f"{cls}/{pdf.name}"
            rows.append({"file": rel, "length_class": cls, "include": "y", "notes": DEFAULT_NOTES.get(rel, "")})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def doc_id_for(row: dict[str, str], prefixes: dict[str, str]) -> str:
    stem = Path(row["file"]).stem
    num = "".join(ch for ch in stem if ch.isdigit())  # full number (was [-2:], which made judgement_101 collide with judgement_01)
    return f"{prefixes[row['length_class']]}{num}"
