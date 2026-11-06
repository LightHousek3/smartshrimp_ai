# SmartShrimp AI

- `classification/`: CVIO YOLO26m-cls checkpoint inference and HTTP API.
- `rag/`: document assistant imported from [phamquoctu-se/rag-al](https://github.com/phamquoctu-se/rag-al), with its original commit history.

Python 3.12, CPU runtime. Local weights and `.env` are required.
Source and third-party terms are recorded in `NOTICE` and `LICENSE`.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -c classification/constraints.txt -r classification/requirements-dev.txt
.venv/Scripts/python.exe -m uvicorn classification.app:create_app --factory --host 127.0.0.1 --port 8080 --workers 1
```

Or run `docker compose -p smartshrimp-classification up --build -d --wait`.
Classification is available at `http://127.0.0.1:8080`; set the backend's
`AI_SERVICE_URL` to this address. Docker maps host port 8080 to container port 8000,
leaving host port 8000 available for the separate RAG service.
Keep `API_KEY` private. `.env` also configures `ALLOW_HISTORICAL_CANDIDATE`,
`BACKGROUND_REMOVAL`, `MAX_IMAGE_PIXELS`, `RATE_LIMIT_PER_MINUTE`, `TORCH_NUM_THREADS`.
Put `cvio-classification.pt` and, when background removal is enabled, `u2net.onnx`
in `classification/models/`. Startup checks both SHA-256 values; no automatic download.

BE calls `POST /v1/diagnose` with `X-API-Key` and 1–5 files in multipart field
`images`, each JPEG/PNG/WEBP, at most 10 MiB. Files must belong to the same observation.
The response contains four class probabilities, their mean result, recommendation,
model version and processing time. Errors return `error_code`, `error_message`,
`request_id`. Readiness: `/v1/ready`; authenticated metadata: `/v1/model/info`;
authenticated API schema: `/openapi.json`. BE owns authorization, images and history.

The supplied CVIO checkpoint is `HISTORICAL_NONFINAL`, not an official proposed
release. Its weights are unchanged; `field_validated=false`. Background removal
is a separate input adapter. Model identity and preprocessing are included in metadata.

Classification checks: `python -m pytest -q`, `python -m ruff check classification`,
`python -m ruff format --check classification`.

## RAG service

Use a separate virtual environment and `.env` inside `rag/`; the two services pin
different FastAPI and Pydantic versions. Run RAG commands from that directory:

```powershell
Set-Location rag
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
# Fill in the database URL, Gemini key and shared internal key before starting.
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Alternatively, from the repository root:
`docker compose -p smartshrimp-rag -f rag/docker-compose.yml up --build -d`.
RAG uses port 8000; classification uses port 8080. Configure the backend with
`RAG_SERVICE_URL=http://localhost:8000` and matching `RAG_INTERNAL_API_KEY`.
Run RAG tests from `rag/` with `.venv/Scripts/python.exe -m pytest -q`.
PostgreSQL integration tests require explicit opt-in and a prepared development DB.

The imported snapshot is `430c5c809352c7bf798af098c6e38b12f9113413` on `main`.
Its README references `migrations/` and `scripts/migrate_documents.py`, but those
files are absent from that snapshot. Obtain the matching schema setup from the
RAG maintainer before starting against a new database. See `rag/NOTICE` for origin
and license metadata; classification's AGPL notice does not relicense this module.

For a fresh clone, first add the source remote:
`git remote add rag-upstream https://github.com/phamquoctu-se/rag-al.git`.
Fetch future changes from `rag-upstream`. Keep the original history:
`git subtree pull --prefix=rag rag-upstream main` (without `--squash`).
