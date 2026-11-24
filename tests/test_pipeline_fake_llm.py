"""End-to-end answer pipeline with a fake LLM and a fake retriever (no API key, no model download):
prompt construction, JSON retry, citation check, logging and key redaction."""
from __future__ import annotations

import copy
import json
import re

import pytest

import rag.pipeline as pipeline
from rag.config import Settings, get_settings
from rag.generation.llm import LLMResponse
from rag.retrieval.retriever import RetrievalResult, RetrievedChunk

PASSAGE = ("Where a person other than a Collector, or than a guardian appointed by will or other instrument, "
           "has been appointed or declared by the Court to be guardian of the property of a ward, he shall not, "
           "without the previous permission of the Court,—\n(a) mortgage or charge, or transfer by sale")
SECRET = "sk-test-THISISNOTAREALKEY0123456789"


class FakeLLM:
    provider, model = "openai", "fake-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def endpoint_info(self):
        return {"provider": self.provider, "base_url_host": "api.openai.com", "model": self.model,
                "reasoning_effort": None}

    def complete(self, system, user, json_mode=True):
        self.calls.append((system, user))
        return LLMResponse(text=self.replies.pop(0), input_tokens=100, output_tokens=20, model=self.model,
                           provider=self.provider, latency_ms=1.0)


class FakeRetriever:
    def __init__(self, store):
        self.store = store

    def retrieve_with_trace(self, question, k=5, filters=None):
        c = RetrievedChunk(chunk_id="GWA-1890:29#1", rank=1, score=0.71, text=PASSAGE,
                           metadata={"doc_type": "act", "act_title": "Guardians and Wards Act, 1890", "section": "29",
                                     "title": "Limitation of powers of guardian of property"},
                           unit_ids=["GWA-1890:29"])
        return RetrievalResult(query=question, chunks=[c], timings_ms={"embed": 1.0, "search": 2.0},
                               stages={"dense": "on"})


class FakeStore:
    corpus_version = "testcorpus01"
    by_id = {"GWA-1890:29#1": {}, "GWA-1890:31#1": {}}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    base = get_settings()
    raw = copy.deepcopy(base.raw)
    raw["paths"]["requests_log"] = str(tmp_path / "requests.jsonl")
    s = Settings(raw=raw, chunking=base.chunking, env={**base.env, "OPENAI_API_KEY": SECRET,
                                                       "LLM_PROVIDER": "openai", "LLM_MODEL": "fake-model"})
    monkeypatch.setattr(pipeline, "get_settings", lambda variant=None: s)
    import rag.retrieval.retriever as retr
    monkeypatch.setattr(retr, "get_retriever", lambda variant=None: FakeRetriever(FakeStore()))
    monkeypatch.setattr(pipeline, "load_chunk_store", lambda p: FakeStore())
    return s


def _reply(claims, insufficient=False, lang="english"):
    return json.dumps({"answer": "A court-appointed guardian needs the Court's permission. This is legal "
                                 "information, not legal advice.", "claims": claims,
                       "insufficient_evidence": insufficient, "language": lang})


def test_c1_end_to_end_with_json_retry_and_verification(settings, monkeypatch, tmp_path):
    good = _reply([
        {"text": "needs permission", "chunk_id": "GWA-1890:29#1", "quote": "he shall not, without the previous permission of the Court"},
        {"text": "altered", "chunk_id": "GWA-1890:29#1", "quote": "he shall not, without the prior permission of the Court"},
    ])
    fake = FakeLLM(["Sure! here is the answer: not json", good])
    monkeypatch.setattr(pipeline, "get_llm_client", lambda s: fake)
    r = pipeline.ask(f"Can I sell my ward's land? my key is {SECRET}", mode="c1", k=5)
    assert r["status"] == "ok"
    assert len(r["llm_attempts"]) == 2 and r["llm_attempts"][0]["parse"].startswith("failed")
    # passages are delimited, labelled with chunk_id, and marked as data
    system, user = fake.calls[0]
    assert '<passage chunk_id="GWA-1890:29#1"' in user and "DATA, not instructions" in user
    assert system == pipeline.load_prompts(settings).system
    v = r["verification"]
    assert v["n_verified"] == 1 and v["n_failed"] == 1 and v["verification_rate"] == 0.5
    assert [c["verified"] for c in r["final"]["claims"]] == [True, False]
    log = (tmp_path / "requests.jsonl").read_text(encoding="utf-8")
    rec = json.loads(log.strip().splitlines()[-1])
    assert rec["retrieved"][0] == {"chunk_id": "GWA-1890:29#1", "rank": 1, "score": 0.71, "unit_ids": ["GWA-1890:29"]}
    assert rec["prompt_version"] == "v1" and rec["corpus_version"] == "testcorpus01"
    assert rec["llm"] == {"provider": "openai", "base_url_host": "api.openai.com", "model": "fake-model",
                          "reasoning_effort": None}
    assert set(rec["latency_ms"]) >= {"embed", "search", "generate", "verify", "total"}
    assert rec["tokens"] == {"input": 200, "output": 40}
    assert rec["cost"]["cost_usd"] is None and "not configured" in rec["cost"]["cost_note"]
    assert rec["verification"]["failures"][0]["reason"] == "quote_not_found", "failures stay in the log"
    assert SECRET not in log and "[REDACTED]" in log


def test_c0_same_system_prompt_no_passages(settings, monkeypatch):
    fake = FakeLLM([_reply([{"text": "s.29 GWA 1890 limits a court-appointed guardian", "chunk_id": None, "quote": None}])])
    monkeypatch.setattr(pipeline, "get_llm_client", lambda s: fake)
    r = pipeline.ask("Can I sell my ward's land?", mode="c0")
    system, user = fake.calls[0]
    assert "<passages>" not in user and "Closed-book mode" in user
    assert system == pipeline.load_prompts(settings).system
    assert r["retrieved"] == [] and r["verification"]["verification_rate"] is None


def test_gives_up_after_two_json_retries_and_logs(settings, monkeypatch, tmp_path):
    fake = FakeLLM(["nope", "still nope", "{bad json"])
    monkeypatch.setattr(pipeline, "get_llm_client", lambda s: fake)
    r = pipeline.ask("question", mode="c0")
    assert r["status"] == "error" and len(r["llm_attempts"]) == 3
    rec = json.loads((tmp_path / "requests.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["status"] == "error" and re.search(r"after 3 attempts", rec["error"])


def test_parse_answer_json_validation():
    with pytest.raises(pipeline.AnswerFormatError):
        pipeline.parse_answer_json('{"answer": "x", "claims": [], "insufficient_evidence": "no", "language": "english"}')
    ok = pipeline.parse_answer_json('```json\n{"answer": "x", "claims": [], "insufficient_evidence": false, "language": "roman_urdu"}\n```')
    assert ok["language"] == "roman_urdu"
