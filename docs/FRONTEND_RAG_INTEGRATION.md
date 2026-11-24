# Frontend integration guide: RAG baseline API

This guide describes the FastAPI contract that exists in `rag/api.py`. The frontend should call this backend over HTTP; it should not connect directly to Chroma, load the embedding model, or call the LLM provider.

## 1. Start the backend locally

From the repository root in PowerShell:

```powershell
.venv\Scripts\python.exe -m uvicorn rag.api:app --port 8000
```

Keep that terminal open while developing. The API base URL is `http://localhost:8000`. Check the interactive API page at `http://localhost:8000/docs`, then check health with `GET http://localhost:8000/health`.

The current CORS allow-list contains exactly `http://localhost:3000`. If the frontend runs on Vite's default `http://localhost:5173` or another origin, either run the frontend on port 3000 or ask the backend owner to add that exact origin to `CORSMiddleware` in `rag/api.py`. Browser requests do not use cookies; CORS credentials are disabled.

## 2. Available endpoints

| Method and path | Purpose |
|---|---|
| `GET /health` | Check service, selected corpus/index and whether an LLM is configured. |
| `POST /ask` | Ask a question in C0 or C1 and receive the answer, claims, retrieval info and timing/token metadata. |
| `GET /chunk?chunk_id=...` | Fetch a complete passage and metadata for a source drawer. Prefer this query form for chunk IDs containing `#`. |
| `GET /chunk/{chunk_id:path}` | Alternative path form for a chunk lookup; URL-encode reserved characters. |
| `GET /docs` | FastAPI's interactive API documentation. |

### Health response

Typical response:

```json
{
  "status": "ok",
  "chunking_variant": "default",
  "collection": "lawfiqh__default",
  "embedding": "bge_m3_local:BAAI/bge-m3",
  "llm_configured": true,
  "llm": "openai_compatible:gemini-3.5-flash-lite",
  "llm_base_url_host": "generativelanguage.googleapis.com",
  "llm_reasoning_effort": null,
  "corpus_version": "38760e62e65b",
  "chunks": 906,
  "indexed_vectors": 906,
  "index_corpus_version": "38760e62e65b"
}
```

The exact corpus version and configured model can change. A healthy process can return HTTP 200 with `status: "degraded"` and a `warning` if its corpus or index is missing or out of sync. Show a degraded state instead of treating every HTTP 200 as ready. Health reports model/provider names, never API keys.

## 3. Ask request

`POST /ask` with `Content-Type: application/json`:

```json
{
  "question": "Can a court-appointed guardian sell a minor's land?",
  "mode": "c1",
  "k": 5
}
```

| Field | Rules | Default |
|---|---|---|
| `question` | Required string, 1–1,000 characters. | — |
| `mode` | `"c1"` or `"c0"`. | `"c1"` |
| `k` | Integer from 1 to 20; number of passages requested for C1. | `5` |

**C1** retrieves passages and gives them to the answer model. **C0** sends no passages (closed-book control); its response has `k: null`, an empty `retrieved` array, and citations are normally null. The endpoint still accepts the same request shape for both modes.

## 4. Successful response shape

The API returns a reduced version of the backend pipeline result. This is an illustrative example; scores, IDs, text and timing values vary per request.

```json
{
  "request_id": "a1b2c3d4e5f6...",
  "mode": "c1",
  "question": "Can a court-appointed guardian sell a minor's land?",
  "k": 5,
  "status": "ok",
  "error": null,
  "final": {
    "answer": "The answer text returned by the model.",
    "claims": [
      {
        "text": "A claim made in the answer.",
        "chunk_id": "GWA-1890:29#1",
        "quote": "The exact supporting words from the passage.",
        "verified": true,
        "reason": "ok"
      }
    ],
    "insufficient_evidence": false,
    "language": "english"
  },
  "verification": {
    "n_claims": 1,
    "n_checked": 1,
    "n_verified": 1,
    "n_failed": 0,
    "verification_rate": 1.0,
    "failures": []
  },
  "retrieved": [
    {
      "chunk_id": "GWA-1890:29#1",
      "rank": 1,
      "score": 0.64,
      "unit_ids": ["GWA-1890:29"]
    }
  ],
  "latency_ms": {
    "embed": 650.0,
    "search": 4.0,
    "generate": 2900.0,
    "verify": 0.2,
    "total": 3600.0
  },
  "tokens": {"input": 2100, "output": 300},
  "cost": {"cost_usd": null, "cost_note": "price not configured"},
  "corpus_version": "38760e62e65b",
  "prompt_version": "v1",
  "chunking_variant": "default"
}
```

Frontend notes:

- Use `final.answer` as the answer body and render `final.claims` separately if you want claim-level citations.
- Each claim's `verified` is `true`, `false`, or `null`. `true` means the normalized quote was found inside that cited passage. It does **not** mean a lawyer verified the legal conclusion. Label it “quote verified” or “quote not found,” not “answer legally verified.”
- `verification_rate` can be `null` when there are no checked citations (typical for C0).
- `insufficient_evidence` is the model's signal. If true, show it clearly and do not hide it; the backend does not convert it into a separate HTTP error.
- `retrieved` includes IDs, rank, similarity score, and parent-unit IDs, but no passage text. Fetch a chunk from `/chunk` when the user opens a citation/source panel.
- `score` is a vector similarity ranking value, **not a probability or confidence percentage**. Do not display it as “64% confidence.”
- Latencies are milliseconds. C0 has no embedding/search stage; those values may be `null`. Token counts and costs may also be `null`.
- The response includes `error` as null on success. Treat `status` as the application-level result indicator as well as checking HTTP status.

## 5. Fetch examples (TypeScript)

```ts
const API_BASE = import.meta.env.VITE_RAG_API_URL ?? "http://localhost:8000";

export type RagMode = "c0" | "c1";

export async function askRag(
  question: string,
  mode: RagMode = "c1",
  k = 5,
) {
  const response = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode, k }),
  });

  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.detail ?? `RAG request failed (${response.status})`);
  }
  if (body?.status !== "ok") {
    throw new Error(body?.error ?? "The answer service could not complete this request.");
  }
  return body;
}

export async function getChunk(chunkId: string) {
  const url = new URL(`${API_BASE}/chunk`);
  url.searchParams.set("chunk_id", chunkId);

  const response = await fetch(url);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.detail ?? `Source lookup failed (${response.status})`);
  }
  return body;
}
```

Vite `.env.local` example:

```text
VITE_RAG_API_URL=http://localhost:8000
```

Only put the backend URL in frontend configuration. The LLM API key belongs in the backend `.env`; never bundle it into frontend JavaScript.

## 6. Source panel response

`GET /chunk?chunk_id=GWA-1890%3A29%231` returns:

```json
{
  "chunk": {
    "chunk_id": "GWA-1890:29#1",
    "parent_id": "GWA-1890:29",
    "unit_ids": ["GWA-1890:29"],
    "doc_id": "GWA-1890",
    "doc_type": "act",
    "unit_type": "section",
    "title": "",
    "act_title": "Guardians and Wards Act, 1890",
    "section": "29",
    "jurisdiction": "Pakistan (federal)",
    "as_of": "TODO",
    "source_file": "data/raw/acts/pdf/GWA-1890.pdf",
    "text": "Full source passage text appears here.",
    "token_count": 300
  },
  "prev_chunk_id": "GWA-1890:28#1",
  "next_chunk_id": "GWA-1890:30#1",
  "notice": "Legal information, not legal advice."
}
```

The `chunk` object is the complete stored record, so fields can be null or lists and may vary by document type. For a statute, useful display fields include `act_title`, `section` or `article`, `title`, `as_of`, `jurisdiction`, and `text`. For a judgment, use fields such as `title` (case name), `case_no`, `court`, `date`, `paragraph`, `jurisdiction`, and `text`. `prev_chunk_id` and `next_chunk_id` can power source navigation; either can be null.

Chunk IDs commonly contain `#` (for example, `GWA-1890:29#1`). Prefer the query-string endpoint and `URL.searchParams` to avoid path-encoding errors. Unknown IDs return HTTP 404.

## 7. Errors and loading states

| HTTP status / response | Frontend behavior |
|---|---|
| `200`, `status: "ok"` | Render answer and claims. |
| `200`, `status: "error"` | Show `error`; this can happen when the model response could not be parsed after retries. |
| `422` | Request validation failed (empty/too-long question, invalid mode, or `k` outside 1–20); show a field-level input error. |
| `503` with `detail` | LLM is not configured; show service-unavailable state. |
| `500` with `detail` | Backend processing failed. Show a retry option and preserve the question. |
| `404` with `detail` on chunk lookup | The source ID is unknown; show a source-unavailable state. |
| `GET /health` returns `status: "degraded"` | Show that the API is up but its corpus/index needs attention. |

LLM-backed answers can take several seconds and provider retries can take longer. Keep a visible loading state, prevent accidental duplicate submissions, and avoid a very short browser timeout. The API has no streaming endpoint, so the answer arrives as one JSON response.

## 8. Suggested UI behavior

1. On app load, call `/health`; show ready/degraded/offline state.
2. Let the user choose **Evidence-backed (C1)** or **Closed-book (C0)**. Default to C1. If exposing `k`, constrain it to 1–20 and default to 5.
3. Submit to `/ask`; show loading state and retain the user's question on error.
4. Render `final.answer`, then claim cards with their exact quote, source ID, and a quote-verification label.
5. Clicking a source ID fetches `/chunk?chunk_id=...` and opens the passage and legal-source metadata.
6. Display `insufficient_evidence` prominently. Keep the legal-information-not-legal-advice notice visible.

## 9. Backend contract limits to account for

- This is a local prototype API with no login, user/session handling, rate limiting, or streaming.
- CORS currently allows only `http://localhost:3000`; the API has no credentialed browser access.
- The response is not a full conversation API: there is no chat history or follow-up state. Send each question as a new `/ask` request.
- C1 uses dense retrieval only. Returned sources are ranking candidates, and their scores are not calibrated confidence.
- Logs are written by the backend and include the question text. Do not place sensitive personal/case information in demo questions unless the deployment's logging policy has been reviewed.

## 10. Backend source references

- `rag/api.py`: CORS, request validation, `/health`, `/ask`, and `/chunk` response contracts.
- `rag/pipeline.py`: C0/C1 execution and answer response fields.
- `rag/generation/verify.py`: meaning of `verified`, `reason`, and `verification_rate`.
- `rag/generation/prompts.py` and `prompts/`: output JSON shape and source-tagged passage handling.
