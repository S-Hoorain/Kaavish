"""Split a judgment's blocks into header, body paragraphs (numbered units) and signature."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.ingest.judgments.pdf_extract import Block, DocLayout, PARA_NUM

# Title line: JUDGMENT / J U D G M E N T / ORDER / O R D E R (not "ORDER SHEET", "Judgment sheet")
TITLE_RE = re.compile(
    r"(?:^|(?<=[\s.\u2026_\-\u2013\u2014*:]))(?<!DATE )(?<!Date )(J\s?U\s?D\s?G\s?E?\s?M\s?E\s?N\s?T|O\s?R\s?D\s?E\s?R)\b\.?"
    r"(?!\s*(?:SHEET|[Ss]heet|WITH|[Ww]ith|dated|DATED|passed|of\b))(?:\s*[\-\u2013\u2014_.\u2026*]{2,})?\s*")
_NAME = r"(?P<name>[A-Z][A-Za-z'’\-]+(?:\s+(?:[A-Z][A-Za-z'’\-]+|\([A-Za-z]+\))){1,6})"
JUDGE_INTRO = re.compile(
    r"^" + _NAME + r"\s*[,.]?\s*[\-–—]?\s*J\s*(?:[.,;:]\s*[\-–—:;,]*|\s+[\-–—]+|(?=\s+[A-Z0-9\[(]))\s*")
SIGNATURE_RE = re.compile(r"^\s*(?:J\s?U\s?D\s?G\s?E|JUDGE|Judge)(?:\s+[A-Z][A-Za-z./]*){0,3}\s*$")
HEADER_HINTS = re.compile(
    r"HIGH COURT|SUPREME COURT|Petition\s+No|\bC\.?\s?P\.?\s*No|\bCP\s*No|Date\s+Order|DATE\s+ORDER|"
    r"For\s+orders|FOR\s+ORDERS|For\s+hearing|Advocate|advocate|A\.A\.G|Advocate\s+General|"
    r"\bpresent\b|Present:|PRESENT|Fresh\s+case|Directions|D/O\s+MATTER|Order\s+Sheet|ORDER\s+SHEET|"
    r"Petitioners?\s*:|Respondents?\s*(?:No\.?\s*\d+)?\s*:|Nemo|Date\s+of|Dates\s+of|Attorney", re.I)
DATE_LINE = re.compile(r"^\s*\d{1,2}[\-./]\d{1,2}[\-./]\d{2,4}\.?\s*:?\s*$")


@dataclass
class ParaUnit:
    label: str                     # "1", "2", "4-5", ... (normalised, no leading zeros)
    blocks: list[Block]
    numbering: str                 # "explicit" | "inferred" | "intro"
    text_override: str | None = None   # first unit: text with the title stripped

    @property
    def text(self) -> str:
        if self.text_override is not None:
            rest = [b.text for b in self.blocks[1:]]
            return "\n".join([self.text_override] + rest).strip()
        return "\n".join(b.text for b in self.blocks).strip()

    @property
    def page_start(self) -> int:
        return self.blocks[0].page_start

    @property
    def markers(self) -> list[str]:
        return [m for b in self.blocks for m in b.markers]


@dataclass
class JudgmentStructure:
    header_texts: list[str]
    title: str | None
    units: list[ParaUnit]
    signature: list[str]
    body_start_method: str
    anomalies: list[str] = field(default_factory=list)
    indented_chars: dict[str, int] = field(default_factory=dict)


def _norm_label(raw: str) -> tuple[str, int, int]:
    nums = [int(x) for x in re.findall(r"\d+", raw)]
    lab = "-".join(str(n) for n in nums)
    return lab, nums[0], nums[-1]


def _is_headerish(b: Block) -> bool:
    t = b.text
    return len(t) < 450 and (bool(HEADER_HINTS.search(t)) or bool(DATE_LINE.match(t)))


def split_structure(layout: DocLayout, blocks: list[Block]) -> JudgmentStructure:
    anomalies: list[str] = []
    left = layout.left_margin
    title = None
    body_start = None
    first_text_override = None
    method = ""
    # 1) a title line (JUDGMENT / ORDER) within the first blocks
    for i, b in enumerate(blocks[:16]):
        m = TITLE_RE.search(b.text)
        if not m:
            continue
        before = b.text[:m.start()].strip()
        # the title must be the block's first word, or follow header material / separators
        if before and not (_is_headerish(b) or re.fullmatch(r"[\W_]*", before) or i > 0 and len(before) < 450):
            continue
        if re.match(r"\s*(SHEET|[Ss]heet)", b.text[m.end():]):
            continue
        title = re.sub(r"\s+", "", m.group(1)).upper()
        rest = b.text[m.end():].strip()
        if rest:
            body_start, first_text_override = i, rest
        else:
            body_start = i + 1
        method = "title"
        header_end = i
        break
    # 2) the judge's opening line ("X, J.:")
    if body_start is None:
        for i, b in enumerate(blocks[:20]):
            if JUDGE_INTRO.match(b.text) and len(b.text) > 80:
                body_start, method, header_end = i, "judge_intro", i
                break
    # 3) order sheets: body starts after the last header-like block
    if body_start is None:
        last = -1
        for i, b in enumerate(blocks[:20]):
            if _is_headerish(b):
                last = i
            elif len(b.text) > 120:
                break
        body_start, method, header_end = last + 1, "after_header", last + 1
        anomalies.append("no title line or judge opening line; body start inferred from header-like blocks")
    header_texts = [b.text for b in blocks[:header_end]]
    if method == "title":
        tb = blocks[header_end].text
        pre = tb[:TITLE_RE.search(tb).start()].strip()
        if pre:
            header_texts.append(pre)
    elif method == "judge_intro":
        pass

    body = blocks[body_start:]
    # signature / trailer
    signature: list[str] = []
    for j, b in enumerate(body):
        if j == 0:
            continue
        near_end = j >= len(body) - 4
        if (SIGNATURE_RE.match(b.text) and len(b.text) < 40) or (near_end and re.search(r"\bJUDGE\b", b.text) and len(b.text) < 220):
            signature = [x.text for x in body[j:]]
            body = body[:j]
            break

    first_override = first_text_override

    # numbering
    def explicit(b: Block):
        if b.is_indented(left):
            return None
        text = b.text
        if first_override is not None and b is body[0]:
            text = first_override
        m = PARA_NUM.match(text)
        return m.group(1) if m else None

    numbered = [explicit(b) for b in body]
    # explicit numbering only when numbered blocks form a real share of the body (not one embedded list)
    n_num = sum(1 for n in numbered if n)
    n_plain = sum(1 for b, n in zip(body, numbered) if not n and not b.is_indented(left))
    any_numbered = n_num >= 3 or (n_num >= 2 and n_num >= 0.3 * (n_num + n_plain))
    units: list[ParaUnit] = []
    indented_chars: dict[str, int] = {}
    last_num = 0
    first_num = next((_norm_label(n)[1] for n in numbered if n), None)
    for b, raw in zip(body, numbered):
        is_ind = b.is_indented(left)
        accepted = False
        if raw and any_numbered:
            lab, lo, hi = _norm_label(raw)
            if (last_num == 0 and lo <= 3) or (last_num < lo <= last_num + 3):
                if lo != last_num + 1 and last_num:
                    anomalies.append(f"numbering jumps from {last_num} to {lab}")
                units.append(ParaUnit(label=lab, blocks=[b], numbering="explicit"))
                last_num = hi
                accepted = True
            else:
                anomalies.append(f"block numbered '{raw}' after paragraph {last_num} treated as part of it "
                                 f"(embedded list or quotation)")
        if accepted:
            continue
        if not units:
            if any_numbered:
                lab = "0" if first_num == 1 else "1"
                units.append(ParaUnit(label=lab, blocks=[b], numbering="intro"))
                last_num = max(last_num, int(lab))
            else:
                units.append(ParaUnit(label="1", blocks=[b], numbering="inferred"))
            continue
        if not any_numbered and not is_ind:
            units.append(ParaUnit(label=str(len(units) + 1), blocks=[b], numbering="inferred"))
            continue
        units[-1].blocks.append(b)  # quotation, list item or continuation of the current paragraph
        if is_ind:
            indented_chars[units[-1].label] = indented_chars.get(units[-1].label, 0) + len(b.text)
    if units and first_override is not None:
        units[0].text_override = first_override
    if not any_numbered:
        anomalies.append("no paragraph numbers in the judgment; paragraphs numbered by position (inferred)")
    # make labels unique
    seen: dict[str, int] = {}
    for u in units:
        if u.label in seen:
            seen[u.label] += 1
            anomalies.append(f"duplicate paragraph label {u.label}")
            u.label = f"{u.label}_{seen[u.label]}"
        else:
            seen[u.label] = 1
    return JudgmentStructure(header_texts=header_texts, title=title, units=units, signature=signature,
                             body_start_method=method, anomalies=anomalies, indented_chars=indented_chars)
