"""Evidence gate, failed-quote retry/drop, and history condensing (fake LLM, fake retriever)."""
from __future__ import annotations

import copy
import json

import pytest

import rag.pipeline as pipeline
from rag.config import Settings, get_settings
from rag.generation import answer_checks as ac
from rag.retrieval.retriever import RetrievalResult, RetrievedChunk
from tests.test_pipeline_fake_llm import PASSAGE, FakeLLM, FakeStore

GOOD_QUOTE = "he shall not, without the previous permission of the Court"
BAD_QUOTE = "he shall not, without the prior permission of the Court"


class ScoredRetriever:
    def __init__(self, score, reranker="on"):
        self.score, self.reranker, self.store, self.queries = score, reranker, FakeStore(), []

    def retrieve_with_trace(self, question, k=5, filters=None):
        self.queries.append(question)
        c = RetrievedChunk(chunk_id="GWA-1890:29#1", rank=1, score=self.score, text=PASSAGE,
                           metadata={"doc_id": "GWA-1890", "doc_type": "act", "act_title": "Guardians and Wards Act, 1890",
                                     "section": "29", "title": "Limitation of powers"}, unit_ids=["GWA-1890:29"])
        return RetrievalResult(query=question, chunks=[c], timings_ms={"embed": 1.0, "search": 1.0},
                               stages={"dense": "on", "reranker": self.reranker})


def reply(claims, insufficient=False, lang="english", answer="A guardian needs permission. Not legal advice."):
    return json.dumps({"answer": answer, "claims": claims, "insufficient_evidence": insufficient, "language": lang})


def claim(quote, text="needs permission"):
    return {"text": text, "chunk_id": "GWA-1890:29#1", "quote": quote}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    def make(score=0.8, reranker="on", answering=None):
        base = get_settings()
        raw = copy.deepcopy(base.raw)
        raw["paths"]["requests_log"] = str(tmp_path / "requests.jsonl")
        raw["answering"] = answering or {}
        s = Settings(raw=raw, chunking=base.chunking, env={**base.env, "LLM_PROVIDER": "openai", "LLM_MODEL": "fake-model",
                                                           "OPENAI_API_KEY": "sk-test"})
        monkeypatch.setattr(pipeline, "get_settings", lambda variant=None: s)
        retr = ScoredRetriever(score, reranker)
        import rag.retrieval.retriever as rmod
        monkeypatch.setattr(rmod, "get_retriever", lambda variant=None: retr)
        monkeypatch.setattr(pipeline, "load_chunk_store", lambda p: FakeStore())
        return retr
    return make


def use_llm(monkeypatch, replies):
    fake = FakeLLM(replies)
    monkeypatch.setattr(pipeline, "get_llm_client", lambda s: fake)
    return fake


GATE = {"evidence_gate": {"enabled": True, "refuse_below": 0.10, "nudge_above": 0.50}}


def test_language_detection():
    assert ac.detect_language("What are visitation rights of a father?") == "english"
    assert ac.detect_language("bachon ki custody aur father ki visitation ka kya qanoon hai") == "roman_urdu"
    assert ac.detect_language("بچوں کی تحویل") == "urdu"


def test_gate_refuses_low_score_without_calling_llm(setup, monkeypatch):
    setup(score=0.03, answering=GATE)
    fake = use_llm(monkeypatch, [])
    r = pipeline.ask("What is the law on inheritance of agricultural land?")
    assert fake.calls == []
    assert r["final"]["insufficient_evidence"] is True and r["final"]["claims"] == []
    assert r["evidence_gate"]["decision"] == "refused_low_score"
    assert r["final"]["searched"] == ["GWA-1890"]


def test_gate_skipped_for_roman_urdu(setup, monkeypatch):
    setup(score=0.03, answering=GATE)
    fake = use_llm(monkeypatch, [reply([claim(GOOD_QUOTE)], lang="roman_urdu")])
    r = pipeline.ask("bachon ki custody ka kya qanoon hai")
    assert len(fake.calls) == 1
    assert r["evidence_gate"]["decision"] == "skipped_roman_urdu"


def test_gate_off_when_reranker_off(setup, monkeypatch):
    setup(score=0.03, reranker="off", answering=GATE)
    use_llm(monkeypatch, [reply([claim(GOOD_QUOTE)])])
    r = pipeline.ask("Can a guardian sell the land?")
    assert r["evidence_gate"]["decision"] == "skipped_no_reranker"


def test_nudge_when_llm_refuses_but_score_high(setup, monkeypatch):
    setup(score=0.85, answering=GATE)
    fake = use_llm(monkeypatch, [reply([], insufficient=True), reply([claim(GOOD_QUOTE)])])
    r = pipeline.ask("Can a guardian sell the ward's land?")
    assert len(fake.calls) == 2 and "rated highly relevant" in fake.calls[1][1]
    assert r["evidence_gate"]["decision"] == "nudged_after_refusal"
    assert r["final"]["insufficient_evidence"] is False and r["verification"]["n_verified"] == 1


def test_failed_quote_retried_then_fixed(setup, monkeypatch):
    setup(answering={"failed_quotes": "retry_then_drop"})
    fake = use_llm(monkeypatch, [reply([claim(BAD_QUOTE)]), reply([claim(GOOD_QUOTE)])])
    r = pipeline.ask("Can a guardian sell the ward's land?")
    assert len(fake.calls) == 2 and BAD_QUOTE in fake.calls[1][1]
    assert r["verification"]["n_failed"] == 0 and r["dropped_claims"] == []
    assert r["verification_first_pass"]["n_failed"] == 1


def test_failed_quote_dropped_when_retry_also_fails(setup, monkeypatch):
    setup(answering={"failed_quotes": "retry_then_drop"})
    ans = "Needs permission. Also invented thing. Not legal advice."
    bad = reply([claim(GOOD_QUOTE, "Needs permission."), claim("not in passage", "Also invented thing.")], answer=ans)
    use_llm(monkeypatch, [bad, bad])
    r = pipeline.ask("Can a guardian sell the ward's land?")
    assert [c["text"] for c in r["final"]["claims"]] == ["Needs permission."]
    assert "Also invented thing." not in r["final"]["answer"]
    assert len(r["dropped_claims"]) == 1


def test_all_claims_failing_becomes_insufficient(setup, monkeypatch):
    setup(answering={"failed_quotes": "retry_then_drop"})
    bad = reply([claim("not in passage")])
    use_llm(monkeypatch, [bad, bad])
    r = pipeline.ask("Can a guardian sell the ward's land?")
    assert r["final"]["claims"] == [] and r["final"]["insufficient_evidence"] is True


def test_keep_policy_is_unchanged_r0(setup, monkeypatch):
    setup(answering={})
    fake = use_llm(monkeypatch, [reply([claim(BAD_QUOTE)])])
    r = pipeline.ask("Can a guardian sell the ward's land?")
    assert len(fake.calls) == 1 and r["final"]["claims"][0]["verified"] is False
    assert "dropped_claims" not in r


def test_history_condensed_before_retrieval(setup, monkeypatch):
    retr = setup(answering={"history": {"enabled": True, "max_turns": 6}})
    standalone = "My ex-husband has a visitation order for our 3 children from 2 years ago; what should I do now?"
    fake = use_llm(monkeypatch, [standalone, reply([claim(GOOD_QUOTE)])])
    hist = [{"role": "user", "content": "My ex has a visitation order for our 3 children from 2 years ago."},
            {"role": "assistant", "content": "Visitation orders are made under the Guardians and Wards Act."}]
    r = pipeline.ask("what should I do now?", history=hist)
    assert retr.queries == [standalone]
    assert "3 children" in fake.calls[0][1] and r["standalone_question"] == standalone
    assert r["llm_attempts"][0]["purpose"] == "condense_history"


def test_history_ignored_when_disabled(setup, monkeypatch):
    retr = setup(answering={})
    use_llm(monkeypatch, [reply([claim(GOOD_QUOTE)])])
    r = pipeline.ask("what should I do now?", history=[{"role": "user", "content": "earlier"}])
    assert retr.queries == ["what should I do now?"] and r["standalone_question"] is None
