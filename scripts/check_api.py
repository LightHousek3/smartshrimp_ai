"""Exercise a real trained artifact through ASGI (no listening port needed)."""

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/shrimpdx-toy-v1.0.0")
    parser.add_argument("--image", default="training/data/toy-v1/test/Healthy/0.png")
    args = parser.parse_args()
    settings = Settings(
        _env_file=None,
        api_key="local-smoke-test-key-32-characters",
        model_dir=args.model_dir,
        allow_synthetic_model=True,
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/health").status_code == 200
        response = client.post(
            "/v1/diagnose",
            headers={"X-API-Key": settings.api_key.get_secret_value()},
            files=[("images", ("shrimp.png", Path(args.image).read_bytes(), "image/png"))],
        )
        assert response.status_code == 200, response.text
        assert len(response.json()["predictions"]) == 4
        print(json.dumps(response.json(), ensure_ascii=True))


if __name__ == "__main__":
    main()
