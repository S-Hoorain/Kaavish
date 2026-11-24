"""Judgment header and paragraph parser on sample PDFs from data/raw/judgments.

Expected values were read by hand from the PDFs (case number, dates, judge, numbering)."""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import ROOT
from rag.ingest.judgments.build import parse_judgment
from rag.ingest.judgments.header import parse_dates

RAW = ROOT / "data" / "raw" / "judgments"
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="run scripts/setup_raw_data.py first")


def _parse(rel: str, doc_id: str):
    pdf = RAW / rel
    ex = pdf.parent / "existing_json" / (pdf.stem + ".json")
    return parse_judgment(pdf, ex, doc_id, rel.split("/")[0])


def test_long_judgment_header_paragraphs_footnotes():
    p = _parse("long/judgement_07.pdf", "SHC-L07")
    h = p["header"]
    assert h["case_numbers"] == ["C.P. No. S-1270 of 2024"]
    assert h["date"] == "2025-08-06"
    assert h["judge"] == "Muhammad Osman Ali Hadi"
    assert h["petitioner"] == "Muhammad Bilal Alvi"
    assert h["court"] == "High Court of Sindh" and h["bench"] == "Karachi"
    labels = [x["label"] for x in p["paragraphs"]]
    assert labels == [str(i) for i in range(1, 18)]
    first = p["paragraphs"][0]["text"]
    assert first.startswith("Muhammad Osman Ali Hadi, J.:")
    assert "26.07.20241" not in first and "26.07.2024" in first, "superscript footnote marker must be removed"
    assert len(p["footnotes"]) == 7
    assert all("C.P. No. S-1270/2024" not in x["text"] for x in p["paragraphs"]), "running header stripped"
    # numbered paragraphs quoted from the statute inside para 10 must not become paragraphs of this judgment
    assert "Power of the Court to make order as to guardianship" in p["paragraphs"][9]["text"]


def test_quoted_numbered_paragraphs_are_not_new_paragraphs():
    p = _parse("long/judgement_05.pdf", "SHC-L05")
    labels = [x["label"] for x in p["paragraphs"]]
    assert labels == [str(i) for i in range(1, 15)]
    assert len(p["footnotes"]) == 18
    assert p["header"]["date"] == "2025-05-07"


def test_unnumbered_order_gets_inferred_numbers_and_page_label_removed():
    p = _parse("short/short_judgement_03.pdf", "SHC-S03")
    assert [x["numbering"] for x in p["paragraphs"]] == ["inferred"] * 5
    assert all("P a g e" not in x["text"] for x in p["paragraphs"])
    h = p["header"]
    assert h["bench"] == "Circuit Court Larkana"
    assert h["case_numbers"] == ["C.P. No. S-196 of 2022"]
    assert h["date"] == "2022-04-07"
    assert p["signature"][0].startswith("Judge")


def test_order_sheet_without_title_and_combined_paragraph_number():
    p = _parse("short/short_judgement_09.pdf", "SHC-S09")
    labels = [x["label"] for x in p["paragraphs"]]
    assert labels[:3] == ["1", "2", "3"] and "4-5" in labels
    assert p["header"]["date"] == "2025-11-07"
    assert "judge" in p["header"]["unparsed"], "missing judge name must be reported, not silently empty"


def test_header_date_parsing_variants():
    assert parse_dates("29.9.2021 & 05.10.2021") == ["2021-09-29", "2021-10-05"]
    assert parse_dates("6th January, 2022") == ["2022-01-06"]
    assert parse_dates("12-07-2016") == ["2016-07-12"]
