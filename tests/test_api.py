"""FastAPI routes: /health, /chunk/{chunk_id}, /ask (pipeline mocked), CORS for localhost:3000."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag.api import app
from rag.config import get_settings

client = TestClient(app)
HAS_CHUNKS = get_settings().chunks_path.exists()


def test_health_always_answers():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] in ("ok", "degraded")


@pytest.mark.skipif(not HAS_CHUNKS, reason="run ingestion first")
def test_chunk_route_with_encoded_hash_and_query_param():
    r = client.get("/chunk/GWA-1890:29%231")
    assert r.status_code == 200
    body = r.json()
    assert body["chunk"]["chunk_id"] == "GWA-1890:29#1" and body["chunk"]["section"] == "29"
    assert body["next_chunk_id"] == "GWA-1890:30#1"
    assert client.get("/chunk", params={"chunk_id": "GWA-1890:29#1"}).status_code == 200
    assert client.get("/chunk/NOPE:1%231").status_code == 404


def test_ask_validates_and_returns_pipeline_result(monkeypatch):
    import rag.pipeline

    monkeypatch.setattr(rag.pipeline, "ask", lambda q, mode, k: {"request_id": "r1", "mode": mode, "status": "ok",
                                                                 "final": {"answer": "x"}, "question": q, "k": k})
    r = client.post("/ask", json={"question": "Talaq ka notice kis ko dena hota hai?", "mode": "c0", "k": 3})
    assert r.status_code == 200 and r.json()["request_id"] == "r1" and r.json()["mode"] == "c0"
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422
    assert client.post("/ask", json={"question": "x", "mode": "c2"}).status_code == 422


def test_cors_allows_local_frontend():
    r = client.options("/health", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
