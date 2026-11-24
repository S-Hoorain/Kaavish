"""Header metadata of a judgment: court and bench, case type and number(s), parties, dates,
judge and counsel. Every field that cannot be parsed is reported in `unparsed` (never silent)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from rag.ingest.judgments.structure import JUDGE_INTRO

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], start=1)}

CITIES = r"KARACHI|HYDERABAD|SUKKUR|LARKANA|LARKANO|MIRPURKHAS|MIRPUR\s*KHAS"
CASE_RE = re.compile(
    r"(?P<type>Constitution(?:al)?\s+Petition|Const\.?\s*Petition|C\.?\s?P\.?|CP)\s*"
    r"(?:No\.?|NO\.?)?\s*(?P<bench>[SD])?\s*[\-\u2013\u2014]?\s*(?P<num>\d{1,5})"
    r"(?:\s*,\s*\d{1,5})*\s*(?:of|OF|/)\s*(?P<year>(?:19|20)\d{2})")
DATE_FIELD_RE = re.compile(
    r"Dates?\s+of\s+(?P<kind>hearing\s+and\s+order|hearing|heairng|decision|judgment|judgement|order|announcement)"
    r"\s*[:\-]*\s*:?\s*(?P<val>(?:\d{1,2}(?:st|nd|rd|th)?[\s.\-/]+(?:\d{1,2}|[A-Za-z]+),?[\s.\-/]+\d{2,4}"
    r"(?:\s*(?:&|and|,)\s*)?)+)", re.I)
DATE_LINE_RE = re.compile(r"^\s*(\d{1,2})[\-./](\d{1,2})[\-./](\d{4})\b")   # order-sheet date at line start


def parse_dates(val: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r"(\d{1,2})(?:st|nd|rd|th)?[\s.\-/]+(\d{1,2}|[A-Za-z]+),?[\s.\-/]+(\d{2,4})", val):
        d, mo, y = m.groups()
        try:
            month = int(mo) if mo.isdigit() else _MONTHS[mo.lower()]
            year = int(y) + (2000 if len(y) == 2 else 0)
            out.append(date(year, month, int(d)).isoformat())
        except (KeyError, ValueError):
            continue
    return out


def human_date(iso: str | None) -> str:
    if not iso:
        return "date unknown"
    y, m, d = (int(x) for x in iso.split("-"))
    return date(y, m, d).strftime("%d %b %Y").lstrip("0")


def _title(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" ,.;:-")
    if s.isupper():
        def cap(w: str) -> str:
            if re.fullmatch(r"\(?[A-Z]\.?\)?|[IVX]+", w):
                return w
            if w.startswith("("):
                return "(" + cap(w[1:])
            return "-".join(p.capitalize() if len(p) > 2 else p.lower() for p in w.split("-"))
        s = " ".join(cap(w) for w in s.split())
    return s


@dataclass
class Header:
    court: str | None = None
    bench: str | None = None
    case_type: str | None = None
    case_numbers: list[str] = field(default_factory=list)
    petitioner: str | None = None
    respondent: str | None = None
    dates: dict[str, list[str]] = field(default_factory=dict)
    date: str | None = None
    date_basis: str | None = None
    judge: str | None = None
    counsel: list[str] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)   # field -> "pdf" | "existing_json"

    @property
    def case_no(self) -> str | None:
        return self.case_numbers[0] if self.case_numbers else None

    @property
    def parties(self) -> str | None:
        if self.petitioner and self.respondent:
            return f"{self.petitioner} v. {self.respondent}"
        return self.petitioner or self.respondent


PARTY_PATTERNS = [
    # [Mst. X v. Y] / (X Vs. Y) / ('X vs. Y')
    re.compile(r"[\[(\u2018'\u201c]\s*(?P<p>[^\[\]()]{3,120}?)\s+(?:v/s\.?|vs\.?|v\.|versus|Vs\.?|VS\.?)\s+(?P<r>[^\[\]()\u2019\u201d]{3,160}?)\s*[\u2019'\u201d]?\s*[\])]"),
    # X ......... Petitioners. Versus Y ........ Respondents.
    re.compile(r"(?P<p>[A-Z][^\n:]{2,120}?)\s*[.\u2026]{3,}\s*Petitioners?\.?\s*(?:Versus|vs\.?|v/s|VS\.?)\s*(?P<r>[^\n]{2,160}?)\s*[.\u2026]{3,}\s*Respondents?", re.S),
    # X ......v/s...... Y
    re.compile(r"(?P<p>(?:Mst\.?|Mr\.?|Mrs\.?|Ms\.?|[A-Z])[^\n:\u2026]{1,100}?)\s*[.\u2026]{2,}\s*(?:[Vv]/[Ss]|[Vv][Ss]?\.?|[Vv]ersus)\s*[.\u2026]{2,}\s*(?P<r>[^\n\-\u2026]{2,120}?)(?=\s*-{3,}|\s*DATE|\s*Date|\s*$)"),
    # C.P. No. S-196/2022. : X vs. Y
    re.compile(r"(?:of\s+\d{4}|/\d{4})\.?\s*:\s*(?P<p>[^\n:]{3,100}?)\s+(?:vs?\.?|v/s|versus)\s+(?P<r>[^\n]{3,160}?)\.?\s*$", re.M),
    # standalone "X vs. Y" line
    re.compile(r"^(?P<p>[A-Z][^:\n]{2,80}?)\s+(?:vs\.?|v/s|v\.|versus|Vs\.?)\s+(?P<r>[^\n]{3,160})$", re.M),
]
PETITIONER_RE = re.compile(r"\bPETITIONERS?\b\s*[:\-]?\s*(?!through|Through|by\b|:)(?P<p>[A-Z][^:\n]{2,80}?)\s*(?:,?\s*[Tt]hrough\b|\s:|$|,\s*Advocate)", re.M | re.I)
RESPONDENT_RE = re.compile(r"\bRESPONDENTS?\b(?:\s*No\.?\s*\d+(?:\s*(?:&|and|,)\s*\d+)*)?\s*[,:\-]?\s*(?!through|Through|by\b|:)(?P<r>[A-Z][^:\n]{2,80}?)\s*(?:,?\s*[Tt]hrough\b|\s:|$|,\s*Advocate|\s+Date)", re.M | re.I)
COUNSEL_LINE = re.compile(r"Advocate|advocate|A\.A\.G|A\.G\.|A\.P\.G|Advocate General|Prosecutor|Attorney|Nemo|None present|in person|present person|through\s+M", re.I)


def _clean_party(s: str) -> str:
    s = CASE_RE.sub("", s)
    s = re.sub(r"^.*?HIGH\s+COURT\s+OF\s+[A-Z]+\s*,?\s*(?:(?:AT|CIRCUIT\s+COURT)\s*,?\s*)*(?:" + CITIES + r")?\s*",
               "", s, flags=re.I | re.S)
    s = re.sub(r"\s+", " ", s).strip(" .,:;-\u2026\u2018\u2019\u201c\u201d'\"")
    s = re.sub(r"\s*[.\u2026]{2,}.*$", "", s)
    return s.strip(" .,:;-")


_BAD_PARTY = re.compile(r"^(?:vs?\b|v/s|versus|through|along|alongwith|a/w|No\.?\s*\d|Mr\.|Ms\.|M/s)|Advocate|advocate", re.I)


def _ok_party(p: str | None, maxlen: int = 170) -> bool:
    return bool(p) and 2 < len(p) < maxlen and not _BAD_PARTY.search(p) and not CASE_RE.search(p)


def parse_header(header_texts: list[str], first_body_text: str | None) -> Header:
    h = Header()
    text = "\n".join(header_texts)
    flat = re.sub(r"\s+", " ", text)

    m = re.search(r"(HIGH\s+COURT\s+OF\s+SINDH|SUPREME\s+COURT\s+OF\s+PAKISTAN|FEDERAL\s+SHARIAT\s+COURT|HIGH\s+COURT\s+OF\s+[A-Z]+)", flat, re.I)
    if m:
        h.court = _title(m.group(1).upper()).replace(" Of ", " of ")
        tail = flat[m.end():m.end() + 80]
        cm = re.search(r"CIRCUIT\s+COURT\s*,?\s*(?:AT\s+)?(" + CITIES + r")", tail, re.I)
        sm = re.search(r"^\s*,?\s*(?:AT\s+)?(" + CITIES + r")", tail, re.I)
        if cm:
            h.bench = "Circuit Court " + _title(cm.group(1).upper())
        elif sm:
            h.bench = _title(sm.group(1).upper())
    seen = []
    for cm in CASE_RE.finditer(flat):
        bench = (cm.group("bench") or "").upper()
        num = f"C.P. No. {bench + '-' if bench else ''}{cm.group('num')} of {cm.group('year')}"
        if num not in seen:
            seen.append(num)
        if h.case_type is None:
            h.case_type = "Constitutional Petition"
    h.case_numbers = seen

    for pat in PARTY_PATTERNS:
        pm = pat.search(text)
        if pm:
            p, r = _clean_party(pm.group("p")), _clean_party(pm.group("r"))
            if _ok_party(p, 120) and _ok_party(r):
                h.petitioner, h.respondent = p, r
                break
    if not h.petitioner:
        pm, rm = PETITIONER_RE.search(text), RESPONDENT_RE.search(text)
        if pm and _ok_party(_clean_party(pm.group("p"))):
            h.petitioner = _clean_party(pm.group("p"))
        if rm and _ok_party(_clean_party(rm.group("r"))):
            h.respondent = _clean_party(rm.group("r"))

    for dm in DATE_FIELD_RE.finditer(flat):
        kind = dm.group("kind").lower().replace("heairng", "hearing").replace("judgement", "judgment")
        kind = re.sub(r"\s+", "_", kind)
        h.dates.setdefault(kind, []).extend(parse_dates(dm.group("val")))
    for line in header_texts:
        lm = DATE_LINE_RE.match(line)
        if lm:
            h.dates.setdefault("order_sheet", []).extend(parse_dates(lm.group(0)))
    for kind in ("decision", "judgment", "order", "announcement", "hearing_and_order", "order_sheet"):
        if h.dates.get(kind):
            h.date, h.date_basis = h.dates[kind][-1], kind
            break

    if first_body_text:
        jm = JUDGE_INTRO.match(first_body_text)
        if jm:
            h.judge = _title(jm.group("name"))
    if not h.judge:
        pm = re.search(r"PRESENT\s*:?\s*(?:MR\.?|Mr\.?)\s*JUSTICE\s+([A-Za-z.\-() ]{3,60}?)(?=\s{2,}|$|\n|,|\s+[A-Z][a-z]+\s+(?:vs|v/s)|\s+C\.?P)", text, re.I | re.M)
        if pm:
            h.judge = _title(pm.group(1))

    splitter = re.compile(r"(?<=\.)\s+(?=(?:Mr|Ms|Mrs|Syed|Rana|M/s)\.?\s)|"
                          r"\s+(?=(?:Petitioners?|Respondents?(?:\s+No\.?\s*[\d&,\s]+)?|Dates?\s+of|For\s+the)\s*:)")
    for line in header_texts:
        for part in splitter.split(line):
            part = re.sub(r"\s+", " ", part or "").strip()
            if part and COUNSEL_LINE.search(part) and len(part) < 300 and not DATE_FIELD_RE.match(part):
                h.counsel.append(part)

    for f in ("court", "bench", "case_no", "parties", "date", "judge", "counsel"):
        v = getattr(h, f)
        if not v:
            h.unparsed.append(f)
    return h
