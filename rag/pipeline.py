"""End-to-end answer pipeline for the two conditions:

    C0: LLM alone (same system prompt and JSON schema, no passages)
    C1: LLM + R0 dense retrieval (top-k passages in delimited tags) + citation check

Every call is appended to logs/requests.jsonl, including failures.
"""
from __future__ import annotations

import json
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

from rag.config import Settings, get_settings
from rag.generation import answer_checks as ac
from rag.generation.cost import estimate_cost
from rag.generation.llm import get_llm_client
from rag.generation.prompts import build_messages, load_prompts
from rag.generation.verify import verify_claims
from rag.index.store import load_chunk_store
from rag.request_log import append_request_log

LANGS = {"english", "urdu", "roman_urdu"}


class AnswerFormatError(ValueError):
    pass


def parse_answer_json(text: str) -> dict[str, Any]:
    """Parse and validate the model's JSON reply. Raises AnswerFormatError with a reason."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    if not t.startswith("{"):
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j <= i:
            raise AnswerFormatError("no JSON object found in the reply")
        t = t[i:j + 1]
    try:
        obj = json.loads(t)
    except json.JSONDecodeError as e:
        raise AnswerFormatError(f"invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise AnswerFormatError("top-level JSON is not an object")
    problems = []
    if not isinstance(obj.get("answer"), str) or not obj["answer"].strip():
        problems.append("'answer' must be a non-empty string")
    claims = obj.get("claims")
    if not isinstance(claims, list):
        problems.append("'claims' must be a list")
    else:
        for i, c in enumerate(claims):
            if not isinstance(c, dict) or not isinstance(c.get("text"), str):
                problems.append(f"claims[{i}] must be an object with a string 'text'")
            elif not all(isinstance(c.get(k), (str, type(None))) for k in ("chunk_id", "quote")):
                problems.append(f"claims[{i}].chunk_id/quote must be strings or null")
    if not isinstance(obj.get("insufficient_evidence"), bool):
        problems.append("'insufficient_evidence' must be true or false")
    if obj.get("language") not in LANGS:
        problems.append(f"'language' must be one of {sorted(LANGS)}")
    if problems:
        raise AnswerFormatError("; ".join(problems))
    return obj


def _retry_note(err: str) -> str:
    return (f"\n\nYour previous reply could not be used: {err}. Reply again with exactly one JSON object "
            f"matching the OUTPUT format in the rules, and nothing else.")


def _generate(llm, system: str, user: str, max_attempts: int, purpose: str,
              attempts: list[dict]) -> dict[str, Any] | None:
    """Call the LLM until the reply parses (JSON retries), appending one record per call to `attempts`."""
    msg = user
    for attempt in range(1, max_attempts + 1):
        resp = llm.complete(system, msg, json_mode=True)
        rec = {"attempt": attempt, "purpose": purpose, "input_tokens": resp.input_tokens,
               "output_tokens": resp.output_tokens, "latency_ms": resp.latency_ms, "model": resp.model,
               "provider_request_id": resp.request_id, "finish_reason": resp.finish_reason, "notes": resp.notes}
        try:
            parsed = parse_answer_json(resp.text)
            rec["parse"] = "ok"
            attempts.append(rec)
            return parsed
        except AnswerFormatError as e:
            rec["parse"] = f"failed: {e}"
            rec["raw_reply"] = resp.text[:4000]
            attempts.append(rec)
            msg = user + _retry_note(str(e))
    return None


def ask(question: str, mode: str = "c1", k: int | None = None, variant: str | None = None,
        log: bool = True, extra_log: dict | None = None,
        history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    s: Settings = get_settings(variant)
    mode = mode.lower()
    if mode not in ("c0", "c1"):
        raise ValueError("mode must be 'c0' or 'c1'")
    k = int(k or s.raw["retrieval"]["default_k"])
    request_id = uuid.uuid4().hex
    t_start = time.perf_counter()
    prompts = load_prompts(s)
    timings: dict[str, float | None] = {"embed": None, "search": None, "generate": None, "verify": None}
    retrieved, stages = [], None
    result: dict[str, Any] = {
        "request_id": request_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question, "mode": mode, "k": k if mode == "c1" else None,
        "chunking_variant": s.chunking.name,
        "config": s.snapshot(), "config_hash": s.snapshot_hash(),
        # endpoint actually used (provider, base URL host only, model id); never the key
        "llm": {"provider": s.llm_provider, "base_url_host": s.llm_base_url_host, "model": s.llm_model,
                "reasoning_effort": s.llm_reasoning_effort or None},
        "prompt_version": prompts.version, "prompt_sha256": prompts.sha256,
    }
    store = None
    acfg = s.raw.get("answering", {}) or {}
    gate_cfg = acfg.get("evidence_gate", {}) or {}
    hist_cfg = acfg.get("history", {}) or {}
    quote_policy = acfg.get("failed_quotes", "keep")
    attempts: list[dict] = []
    llm = None
    try:
        # 3. follow-ups: condense history + latest message into one standalone question
        working_q = question
        if history and hist_cfg.get("enabled"):
            llm = get_llm_client(s)
            t_c = time.perf_counter()
            try:
                resp = llm.complete(ac.CONDENSE_SYSTEM,
                                    ac.condense_user_message(question, history, int(hist_cfg.get("max_turns", 6))),
                                    json_mode=False)
                working_q = ac.clean_standalone(resp.text, question)
                attempts.append({"attempt": 1, "purpose": "condense_history", "input_tokens": resp.input_tokens,
                                 "output_tokens": resp.output_tokens, "latency_ms": resp.latency_ms,
                                 "model": resp.model, "provider_request_id": resp.request_id,
                                 "finish_reason": resp.finish_reason, "notes": resp.notes, "parse": "text"})
            except Exception as e:  # a failed rewrite must not break the answer: fall back to the raw message
                result["condense_error"] = f"{type(e).__name__}: {e}"
            timings["condense"] = round((time.perf_counter() - t_c) * 1000, 1)
            result["history_turns"] = len(history)
        result["standalone_question"] = working_q if working_q != question else None
        language = ac.detect_language(question)

        if mode == "c1":
            from rag.retrieval.retriever import get_retriever

            rr = get_retriever(variant).retrieve_with_trace(working_q, k=k)
            retrieved, stages = rr.chunks, rr.stages
            timings.update(rr.timings_ms)
            store = get_retriever(variant).store
        else:
            try:
                store = load_chunk_store(str(s.chunks_path))
            except FileNotFoundError:
                store = None
        result["corpus_version"] = store.corpus_version if store else None
        result["retrieval_stages"] = stages
        result["retrieved"] = [{"chunk_id": c.chunk_id, "rank": c.rank, "score": c.score,
                                "unit_ids": c.unit_ids} for c in retrieved]

        # 1a. evidence gate: refuse before calling the LLM when even the best passage is irrelevant
        gate_active = (mode == "c1" and gate_cfg.get("enabled") and (stages or {}).get("reranker") == "on"
                       and language != "roman_urdu")
        top_score = retrieved[0].score if retrieved else 0.0
        gate: dict[str, Any] = {"active": bool(gate_active), "top_score": top_score if mode == "c1" else None,
                                "language": language, "decision": "not_applied"}
        if mode == "c1" and gate_cfg.get("enabled") and not gate_active:
            gate["decision"] = "skipped_roman_urdu" if language == "roman_urdu" else "skipped_no_reranker"
        result["evidence_gate"] = gate

        parsed = None
        if gate_active and top_score < float(gate_cfg.get("refuse_below", 0.10)):
            gate["decision"] = "refused_low_score"
            searched = sorted({(c.metadata or {}).get("doc_id") or c.chunk_id.split(":")[0] for c in retrieved})
            parsed = ac.refusal_answer(language, searched)
            result["llm_attempts"], result["tokens"] = [], {"input": 0, "output": 0}
            result["cost"] = None
            claims, summary = [], {"n_claims": 0, "n_checked": 0, "n_verified": 0, "n_failed": 0,
                                   "verification_rate": None, "failures": []}
            parsed["claims"] = claims
            result["final"], result["verification"], result["status"] = parsed, summary, "ok"
        else:
            system, user = build_messages(prompts, mode, working_q, retrieved)
            llm = llm or get_llm_client(s)
            result["llm"] = llm.endpoint_info()
            t_gen = time.perf_counter()
            max_attempts = 1 + int(s.raw["llm"]["json_parse_retries"])
            parsed = _generate(llm, system, user, max_attempts, "answer", attempts)

            # 1b. LLM refused although the best passage scored high: ask once more
            if (parsed is not None and gate_active and parsed.get("insufficient_evidence")
                    and not parsed.get("claims") and top_score >= float(gate_cfg.get("nudge_above", 0.50))):
                gate["decision"] = "nudged_after_refusal"
                again = _generate(llm, system, user + ac.nudge_note(top_score), max_attempts, "nudge", attempts)
                if again is not None:
                    parsed = again
            elif gate_active:
                gate["decision"] = "passed"
            timings["generate"] = round((time.perf_counter() - t_gen) * 1000, 1)

            if parsed is None:
                result["status"] = "error"
                result["error"] = f"model reply was not valid JSON after {max_attempts} attempts"
                result["final"] = None
            else:
                t_ver = time.perf_counter()
                context = {c.chunk_id: c.text for c in retrieved}
                corpus = store.by_id if store else None
                claims, summary = verify_claims(parsed.get("claims") or [], context, corpus=corpus, mode=mode)
                parsed["claims"] = claims

                # 2. failed quotes: regenerate once with the failures listed, then drop what still fails
                if mode == "c1" and quote_policy == "retry_then_drop" and summary["n_failed"]:
                    result["verification_first_pass"] = summary
                    again = _generate(llm, system, user + ac.failed_quotes_note(summary["failures"]),
                                      max_attempts, "fix_quotes", attempts)
                    if again is not None:
                        claims2, summary2 = verify_claims(again.get("claims") or [], context, corpus=corpus, mode=mode)
                        if summary2["n_failed"] <= summary["n_failed"]:
                            again["claims"] = claims2
                            parsed, summary = again, summary2
                    parsed, dropped = ac.drop_failed(parsed)
                    result["dropped_claims"] = [{"text": c.get("text"), "chunk_id": c.get("chunk_id"),
                                                 "quote": c.get("quote"), "reason": c.get("reason")} for c in dropped]
                timings["verify"] = round((time.perf_counter() - t_ver) * 1000, 2)
                result["final"] = parsed
                result["verification"] = summary
                result["status"] = "ok"

            in_tok = sum(a["input_tokens"] or 0 for a in attempts) if all(a["input_tokens"] is not None for a in attempts) else None
            out_tok = sum(a["output_tokens"] or 0 for a in attempts) if all(a["output_tokens"] is not None for a in attempts) else None
            result["llm_attempts"] = attempts
            result["tokens"] = {"input": in_tok, "output": out_tok}
            result["cost"] = estimate_cost(llm.provider, llm.model, in_tok, out_tok)
    except Exception as e:  # logged, then re-raised to the caller with the request id
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc(limit=6)
        raise_after = e
    else:
        raise_after = None
    finally:
        timings["total"] = round((time.perf_counter() - t_start) * 1000, 1)
        result["latency_ms"] = timings
        if extra_log:
            result["extra"] = extra_log
        if log:
            append_request_log(s, result)
    if raise_after is not None:
        raise RuntimeError(f"[request {request_id}] {result['error']}") from raise_after
    return result
