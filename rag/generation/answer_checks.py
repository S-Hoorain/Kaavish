"""Answer-quality steps around generation (all switchable in config `answering:`; all OFF by default,
so R0 behaviour is unchanged unless turned on).

1. evidence_gate   - uses the reranker score of the best passage:
                     below `refuse_below` -> answer "insufficient evidence" without calling the LLM;
                     LLM refused but best score >= `nudge_above` -> ask the LLM once more to answer the
                     supported parts. Only when the reranker is on, and skipped for Roman Urdu questions
                     (the reranker scores Roman Urdu near zero, so its score is not meaningful there).
2. failed_quotes   - "keep" (R0: failures stay, marked verified=false) or "retry_then_drop": regenerate once
                     with the failed quotes listed; claims that still fail are removed (and their sentence
                     removed from the answer when it appears there word for word).
3. history         - condense a follow-up plus earlier turns into one standalone question before retrieval.
"""
from __future__ import annotations

import re
from typing import Any

URDU_SCRIPT = re.compile(r"[؀-ۿ]")
ROMAN_URDU_WORDS = {
    "hai", "hain", "ka", "ki", "ke", "kya", "kia", "mein", "main", "mai", "aur", "ko", "se", "nahi", "nahin",
    "kar", "karna", "karein", "kaise", "kese", "mera", "meri", "mere", "tha", "thi", "hum", "ham", "apne",
    "bachon", "bachay", "bacha", "bache", "qanoon", "qanun", "shohar", "biwi", "chahiye", "sakti", "sakta",
    "liye", "agar", "lekin", "wala", "wali", "abhi", "kyun", "kyon", "talaq", "khula", "nikah",
}


def detect_language(text: str) -> str:
    """english | urdu | roman_urdu (simple heuristic; good enough to decide whether to trust reranker scores)."""
    if URDU_SCRIPT.search(text or ""):
        return "urdu"
    words = re.findall(r"[a-z]+", (text or "").lower())
    hits = sum(1 for w in words if w in ROMAN_URDU_WORDS)
    if words and (hits >= 2 or hits / len(words) >= 0.15):
        return "roman_urdu"
    return "english"


REFUSAL = {
    "english": ("Insufficient evidence: the knowledge base does not contain passages that answer this question. "
                "This is legal information, not legal advice; please consult a lawyer about your situation."),
    "urdu": ("ناکافی شواہد: معلوماتی ذخیرے میں اس سوال کا جواب دینے والا مواد موجود نہیں۔ "
             "یہ قانونی معلومات ہے، قانونی مشورہ نہیں؛ اپنے معاملے کے لیے وکیل سے رجوع کریں۔"),
    "roman_urdu": ("Kaafi saboot nahi: knowledge base mein is sawal ka jawab dene wala mawad maujood nahi. "
                   "Yeh qanooni maloomat hai, qanooni mashwara nahi; apne maamle ke liye wakeel se rujoo karein."),
}


def refusal_answer(language: str, searched: list[str]) -> dict[str, Any]:
    text = REFUSAL.get(language, REFUSAL["english"])
    return {"answer": text, "claims": [], "insufficient_evidence": True, "language": language,
            "searched": searched}


def nudge_note(top_score: float) -> str:
    return (f"\n\nNote: the best passage above was rated highly relevant to the question (score {top_score:.2f}). "
            "Re-read the passages and answer the parts of the question they support, with exact quotes. "
            "Set insufficient_evidence to true only for the parts they genuinely do not cover.")


def failed_quotes_note(failures: list[dict[str, Any]]) -> str:
    lines = "\n".join(f'- chunk_id {f.get("chunk_id")!s}: "{(f.get("quote") or "")[:200]}" ({f.get("reason")})'
                      for f in failures)
    return ("\n\nYour previous reply had quotes that are NOT exact copies of the cited passage:\n" + lines +
            "\nReply again with the full JSON. For each claim, copy the quote character for character from the "
            "passage with that chunk_id, or remove the claim if no passage supports it.")


def drop_failed(parsed: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Remove claims with verified == False. Their text is also removed from the answer when it appears there
    verbatim. If claims existed and none survive, the answer becomes insufficient evidence."""
    claims = parsed.get("claims") or []
    keep = [c for c in claims if c.get("verified") is not False]
    dropped = [c for c in claims if c.get("verified") is False]
    if not dropped:
        return parsed, []
    out = dict(parsed)
    answer = out.get("answer") or ""
    for c in dropped:
        t = (c.get("text") or "").strip()
        if t and t in answer:
            answer = answer.replace(t, "").strip()
    out["answer"] = re.sub(r"\s{2,}", " ", answer).strip()
    out["claims"] = keep
    if claims and not keep:
        out["insufficient_evidence"] = True
        out["answer"] = REFUSAL.get(out.get("language"), REFUSAL["english"])
    return out, dropped


CONDENSE_SYSTEM = (
    "You rewrite a user's latest message in a legal-information chat into ONE standalone question that can be "
    "understood without the conversation. Keep every fact from earlier user turns that the latest message "
    "depends on (who, children, ages, what was ordered, by which court, when). Keep the user's language and "
    "script (English, Urdu, or Roman Urdu). Do not answer the question. Do not add facts. "
    "Reply with the standalone question only, at most 900 characters."
)


def condense_user_message(question: str, history: list[dict[str, str]], max_turns: int) -> str:
    turns = [t for t in (history or []) if t.get("role") in ("user", "assistant") and (t.get("content") or "").strip()]
    turns = turns[-max_turns:]
    convo = "\n".join(f'{t["role"].upper()}: {t["content"].strip()[:1500]}' for t in turns)
    return (f"<conversation>\n{convo}\n</conversation>\n\n<latest_user_message>\n{question}\n</latest_user_message>\n\n"
            "Write the standalone question.")


def clean_standalone(text: str, fallback: str) -> str:
    t = (text or "").strip().strip('"').strip()
    t = re.sub(r"^(standalone question\s*:\s*)", "", t, flags=re.I).strip()
    if not t:
        return fallback
    return t[:1000]
