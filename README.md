# Pakistan Family Law RAG baseline

A research prototype for retrieving passages from Pakistani family-law sources and generating answers with source-linked claims. The repository contains the RAG implementation, evaluation harness, tests, and a FastAPI interface for a separate frontend.

## Repository layout

- `rag/`: ingestion, chunking, embeddings, vector-store access, retrieval, generation, citation checks, evaluation metrics, CLI, and API.
- `config/`, `prompts/`: runtime settings, document/chunking variants, pricing, and versioned prompts.
- `scripts/`: environment check, raw-data setup, and provider connectivity helper.
- `eval/`: evaluation runner and pilot question set.
- `tests/`: unit and API tests.
- `Code/`: reviewed Constitution parser and its source/structured output. Other statutes and judgment documents are not included in this public repository.
- `docs/FRONTEND_RAG_INTEGRATION.md`: concrete frontend API contract and examples.

## Requirements

- Python 3.11 or later (3.12 recommended).
- An LLM provider API key for answer generation; local embedding and retrieval can be used independently.
- The source corpus. The public repository includes Constitution inputs only. The family-law Acts and judgment corpus used by the full baseline is intentionally supplied separately and must be placed in the paths expected by `scripts/setup_raw_data.py` before building the full index.

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

Provide the complete corpus under `Code/` using the folder and filenames in `scripts/setup_raw_data.py`, then copy it into the local raw-data area:

```powershell
python scripts/setup_raw_data.py
```

The setup script reports missing source files. With only the public Constitution files, full family-law ingestion/indexing and answer evaluation are not reproducible: the Acts and judgment source corpus is not included. Once the complete corpus is available, follow the ingestion and index commands exposed by `python -m rag.cli --help` and the settings in `config/`.

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
