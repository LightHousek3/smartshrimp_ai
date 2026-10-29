# SmartShrimp AI

- `classification/`: CVIO YOLO26m-cls checkpoint inference and HTTP API.
- `rag/`: reserved for the document assistant.

Python 3.12, CPU runtime. Local weights and `.env` are required.
Source and third-party terms are recorded in `NOTICE` and `LICENSE`.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -c classification/constraints.txt -r classification/requirements-dev.txt
.venv/Scripts/python.exe -m uvicorn classification.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Or run `docker compose -p smartshrimp-classification up --build -d --wait`.
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

Checks: `python -m pytest -q`, `python -m ruff check .`, `python -m ruff format --check .`.
