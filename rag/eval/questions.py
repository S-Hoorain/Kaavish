"""Benchmark question file loader (eval/pilot_questions.csv)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ["question_id", "question", "language", "gold_chunk_ids", "question_type"]


@dataclass
class Question:
    question_id: str
    question: str
    language: str
    gold: set[str]
    question_type: str
    label_status: str = ""


def load_questions(path: Path) -> list[Question]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing} (FR-EVAL-01: invalid file rejected)")
        out = []
        for i, row in enumerate(reader, start=2):
            if not (row.get("question_id") and row.get("question")):
                raise ValueError(f"{path}:{i}: empty question_id or question")
            gold = {g.strip() for g in (row.get("gold_chunk_ids") or "").split(";") if g.strip()}
            out.append(Question(row["question_id"].strip(), row["question"].strip(), row["language"].strip(),
                                gold, row["question_type"].strip(), (row.get("label_status") or "").strip()))
    return out
