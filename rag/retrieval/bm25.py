"""BM25 keyword index over the chunk store (R1, FR-RET-03).

Plain Okapi BM25, no extra dependency. Built in memory from chunks.jsonl when first needed
(a few thousand chunks build in well under a second).

Why: dense embeddings match meaning but miss exact legal words (section numbers, "hizanat",
"G&W", case numbers). BM25 matches the exact words; reciprocal rank fusion combines both lists.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# \w is Unicode-aware in Python, so this keeps Urdu-script words as well as English / Roman Urdu.
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if len(t) > 1 or t.isdigit()]


class BM25Index:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tfs: list[Counter] = []
        self.lens: list[int] = []
        df: Counter = Counter()
        for cid, text in docs.items():
            toks = tokenize(text)
            tf = Counter(toks)
            self.ids.append(cid)
            self.tfs.append(tf)
            self.lens.append(len(toks))
            df.update(tf.keys())
        n = len(self.ids)
        self.avgdl = (sum(self.lens) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, n: int) -> list[tuple[str, float]]:
        q = [t for t in tokenize(query) if t in self.idf]
        if not q:
            return []
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avgdl or 1))
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scores.append((self.ids[i], s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank of d in that list)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
