# SmartShrimp AI

AI services used by the SmartShrimp backend:

- **Classification:** predicts one of four shrimp health classes using the supplied CVIO YOLO26m-cls checkpoint. This module runs inference; it does not train the model.
- **RAG:** answers questions using indexed documents and Google Gemini. It was imported from [rag-al](https://github.com/phamquoctu-se/rag-al) with its original commit history.

The backend handles user permissions, image storage and request history. The app and web client call the backend, not these services directly.

## Project structure

```text
smartshrimp_ai/
├── classification/       # Classification API, dependencies and tests
│   ├── models/           # Local checkpoint files, not committed
│   └── .env              # Private classification settings
├── rag/                  # RAG API, ingestion code, dependencies and tests
│   └── .env              # Private RAG settings
├── docker-compose.yml    # Runs both services under smartshrimp-ai
├── NOTICE                # Source references and third-party terms
└── LICENSE
```

Each service has its own dependencies and container. There is no root `.env`.

## 1. Prepare local files

Install Git and Docker Desktop with Docker Compose. Clone this repository and open a terminal in its root directory. Python is only needed if you run the services outside Docker.

Ask the team for the approved model files and database configuration. These files are not downloaded automatically:

| File or resource                               | Required for                                                          |
| ---------------------------------------------- | --------------------------------------------------------------------- |
| `classification/models/cvio-classification.pt` | Classification; use the supplied checkpoint, renamed to this filename |
| `classification/models/u2net.onnx`             | Only when `BACKGROUND_REMOVAL=true`                                   |
| PostgreSQL with pgvector and the RAG schema    | RAG startup, document indexing and retrieval                          |
| Gemini API key                                 | RAG embeddings and answers                                            |

Classification checks the model file hashes at startup. Renaming another checkpoint will not make it compatible. The supplied checkpoint is a historical candidate, not an official final release; see `NOTICE` for its identity.

The imported RAG snapshot does not contain the migrations referenced in its README. Get the matching schema setup from the RAG maintainer before using a new database. For answers based on project documents, the database also needs indexed documents.

## 2. Configure each service

Create the following files locally. Replace every `<...>` value before starting. Do not commit credentials or model files.

**`classification/.env`**

```dotenv
API_KEY=<private-key-with-at-least-32-ASCII-characters>
ALLOW_HISTORICAL_CANDIDATE=true
BACKGROUND_REMOVAL=false
MAX_IMAGE_PIXELS=20000000
RATE_LIMIT_PER_MINUTE=60
TORCH_NUM_THREADS=2
```

`ALLOW_HISTORICAL_CANDIDATE=true` explicitly enables the supplied candidate checkpoint. `BACKGROUND_REMOVAL=false` passes the image to native model preprocessing without U²-Net. Set it to `true` only if you want that input adapter and have the approved `u2net.onnx` file.

**`rag/.env`**

```dotenv
APP_ENV=development
INTERNAL_API_KEY=<private-shared-key>
GEMINI_INGESTION_API_KEY=<your-ingestion-api-key>
GEMINI_QUERY_API_KEY=<your-query-api-key>
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2
GEMINI_LLM_MODEL=gemini-3.5-flash-lite
GEMINI_FALLBACK_MODELS=["gemini-3.1-flash-lite"]
DATABASE_URL=<verified-postgresql-connection-url>
```

Use the embedding configuration that matches the existing index. The current code produces 768-dimensional vectors. If Docker cannot reach an IPv6-only Supabase database address, use the project's verified session pooler URL on port `5432`. Obtain that URL from the team or Supabase's Connect dialog.

## 3. Run with Docker

Run these commands from the repository root:

```powershell
docker compose config --quiet
docker compose up --build -d --wait --wait-timeout 180
docker compose ps
```

Both containers should show `healthy` under the `smartshrimp-ai` project.

| Service        | Local URL               | Health check     | Request authentication |
| -------------- | ----------------------- | ---------------- | ---------------------- |
| Classification | `http://localhost:8080` | `GET /v1/ready`  | `X-API-Key`            |
| RAG            | `http://localhost:8000` | `GET /v1/health` | `X-Internal-Key`       |

Useful commands:

```powershell
docker compose logs -f classification rag
docker compose up --build -d --wait classification
docker compose up --build -d --wait rag
docker compose stop
docker compose down
```

The individual `up` commands start or update one service. After changing its `.env`, run `docker compose up -d --wait <service>`. Model files, documents and logs remain on the host when containers are removed. Stop a locally running service before starting Docker on the same port.

## 4. Connect the backend

Set these values in `smartshrimp_be/.env`:

```dotenv
AI_SERVICE_URL=http://localhost:8080
AI_API_KEY=<same-value-as-classification-API_KEY>
RAG_SERVICE_URL=http://localhost:8000
RAG_INTERNAL_API_KEY=<same-value-as-rag-INTERNAL_API_KEY>
```

These URLs assume the backend runs on the host. A backend container needs a reachable Docker hostname and network instead of `localhost`.

**Classification:** `POST /v1/diagnose` accepts 1–5 JPEG, PNG or WebP files in the multipart field `images`. Each file must be at least 224 × 224 pixels and at most 10 MiB. Send images from the same observation together. The response includes the top class, confidence, all four probabilities, recommendation, model version and processing time.

The class identifiers are `Healthy`, `BG`, `WSSV` and `WSSV_BG`. For several images, the classifier averages their class probabilities. The backend applies the recognition policy to the highest mean probability: below 80% is `UNRECOGNIZED`, while 80% or above is `RECOGNIZED`. An unrecognized result has a null public `predictedLabel`; the original four probabilities remain available. This is a confidence rule, not a shrimp detector or a guarantee that the image belongs to a supported class.

**RAG:** `POST /v1/chat` accepts a question and user context from the backend. It returns the answer, retrieved sources and query status. Local API documentation is available at `http://localhost:8000/docs` when `APP_ENV` is not `production`. See [rag/README.md](rag/README.md) for ingestion and request details.

## Local development and tests

Both Docker images and CI jobs use Python 3.12.12. Keep separate virtual environments because the services have different dependency versions. The commands below use PowerShell and the Windows Python launcher. On macOS or Linux, use `python3.12` to create the environment and `.venv/bin/python` to run it.

From the repository root, open one terminal for classification:

```powershell
Set-Location classification
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -c constraints.txt -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m ruff format --check .
.venv/Scripts/python.exe -m uvicorn classification.app:create_app --app-dir .. --factory --host 127.0.0.1 --port 8080 --workers 1
```

Open another terminal at the repository root for RAG:

```powershell
Set-Location rag
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Classification's checkpoint test skips when the local weights are missing. RAG database integration tests require explicit opt-in and a prepared development database. CI runs the two modules in separate jobs.

## Source and licenses

Classification reuses the supplied checkpoint and the native inference workflow from [CVIO](https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26), with integration details in [NOTICE](NOTICE). Classification integration is covered by AGPL-3.0; see [LICENSE](LICENSE) and [third-party notices](classification/THIRD_PARTY_NOTICES).

RAG comes from `phamquoctu-se/rag-al` at commit `430c5c809352c7bf798af098c6e38b12f9113413`. Its imported snapshot has no license file; classification's license does not relicense it. See [rag/NOTICE](rag/NOTICE).
