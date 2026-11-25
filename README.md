# Pakistan Family Law RAG baseline

A research prototype for retrieving passages from Pakistani family-law sources and generating answers with source-linked claims. The repository contains the RAG implementation, evaluation harness, tests, and a FastAPI interface for a separate frontend.

## Repository layout

- `rag/`: ingestion, chunking, embeddings, vector-store access, retrieval, generation, citation checks, evaluation metrics, CLI, and API.
- `config/`, `prompts/`: runtime settings, document/chunking variants, pricing, and versioned prompts.
- `scripts/`: environment check, raw-data setup, and provider connectivity helper.
- `eval/`: evaluation runner and pilot question set.
- `tests/`: unit and API tests.
- `Code/`: reviewed Constitution parser and its source/structured output. Runtime Constitution, Act, and judgment inputs are under `data/raw/`.
- `docs/FRONTEND_RAG_INTEGRATION.md`: concrete frontend API contract and examples.

## Requirements

- Python 3.11 or later (3.12 recommended).
- An LLM provider API key for answer generation; local embedding and retrieval can be used independently.
- The Constitution, four family-law Acts, and judgment source inputs are included under `data/raw/`.

## Set up

PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally and set the provider/API key and model settings required by `config/settings.yaml`. Never commit `.env` or put a provider key in frontend code.

Check the environment:

```powershell
python scripts/check_env.py
```

Build the included Constitution and Acts corpus:

```powershell
python -m rag.cli ingest
python -m rag.cli index
```

The 30 judgment rows in `data/judgment_manifest.csv` are enabled. Ingestion parses the source PDFs and writes parsed outputs under `data/processed/judgments_parsed/`; this generated folder is ignored by Git and must be recreated by running ingestion.

## Run the API

After the corpus has been ingested and the configured vector index has been built:

```powershell
python -m uvicorn rag.api:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000/docs`. The main frontend endpoints are `GET /health`, `POST /ask`, and `GET /chunk?chunk_id=...`. The health endpoint reports whether a usable corpus/index and LLM configuration are present. See [the frontend integration guide](docs/FRONTEND_RAG_INTEGRATION.md) for request/response fields, examples, errors, loading states, and CORS setup.

## Evaluation and tests

The evaluation harness and pilot questions are in `eval/`. Provider-backed evaluation requires the complete corpus, built index, valid API configuration, and network access to the configured provider. Do not treat a partial/API-failed run as a completed benchmark. Evaluation result files and local logs are generated artifacts and are not committed.

Run the test suite with:

```powershell
python -m pytest
```

## Constitution parser

The reviewed source is in `Code/scrape_constitution.py`; its input PDF and structured JSONL are alongside it. Run it from the repository root with:

```powershell
python Code/scrape_constitution.py
```

It writes regenerated extraction outputs under `Code/data/`. To use another PDF or output directory, pass `--pdf` and `--out`.
