import argparse
import io
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps

from .background import BackgroundRemover
from .constants import CHECKPOINT_SHA256, CLASSES
from .model import configure_runtime, sha256, verify_checkpoint
from .settings import ROOT, Settings


def check(args):
    args.output.mkdir(parents=True, exist_ok=False)
    settings = Settings()
    verify_checkpoint(settings.model_dir / "cvio-classification.pt")
    rows = json.loads(args.samples.read_text())
    if len(rows) != 5 or any(row["split"] != "val" for row in rows):
        raise ValueError("Provide five fixed validation samples")
    sources = {}
    for domain in ("processed", "raw"):
        sources[domain] = [Path(row[domain]) for row in rows]
        for row, path in zip(rows, sources[domain], strict=True):
            if sha256(path) != row[domain + "_sha256"]:
                raise ValueError("Qualification image checksum mismatch")
    torch, YOLO = configure_runtime(settings)
    native = YOLO(str(settings.model_dir / "cvio-classification.pt"), task="classify")
    remover = BackgroundRemover(settings.model_dir / "u2net.onnx", settings.torch_num_threads)
    original_inputs = []
    for path in sources["raw"]:
        with Image.open(path) as image:
            original_inputs.append(remover.remove(ImageOps.exif_transpose(image)))
    expected = {}
    for domain, source in (
        ("processed", [str(p) for p in sources["processed"]]),
        ("raw", original_inputs),
    ):
        expected[domain] = np.array(
            [
                r.probs.data.detach().cpu().numpy()
                for r in native.predict(
                    source=source, imgsz=224, device="cpu", half=False, save=False, verbose=False
                )
            ],
            dtype=np.float64,
        )
    key = settings.api_key.get_secret_value() if args.base_url else secrets.token_hex(32)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = args.base_url or f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "API_KEY": key,
        "ALLOW_HISTORICAL_CANDIDATE": "true",
        "BACKGROUND_REMOVAL": "true",
    }
    results = []
    with (args.output / "server.log").open("w", encoding="utf-8") as log:
        server = (
            None
            if args.base_url
            else subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "classification.app:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--workers",
                    "1",
                    "--no-access-log",
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
        try:
            with httpx.Client(base_url=base, timeout=120, trust_env=False) as client:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if server is not None and server.poll() is not None:
                        raise RuntimeError("Server failed to start; inspect server.log")
                    try:
                        if client.get("/v1/ready").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.25)
                else:
                    raise RuntimeError("Readiness timeout")
                headers = {"X-API-Key": key}
                assert client.get("/v1/model/info").status_code == 401
                response = client.get("/v1/model/info", headers=headers)
                assert response.status_code == 200
                info = response.json()
                assert info["checkpoint_sha256"] == CHECKPOINT_SHA256
                assert (
                    info["upstream_status"] == "HISTORICAL_NONFINAL"
                    and info["field_validated"] is False
                )
                for domain, media in (("processed", "image/png"), ("raw", "image/jpeg")):
                    for count in (1, 2, 5):
                        files = [
                            ("images", (p.name, p.read_bytes(), media))
                            for p in sources[domain][:count]
                        ]
                        started = time.perf_counter()
                        response = client.post("/v1/diagnose", headers=headers, files=files)
                        elapsed = time.perf_counter() - started
                        assert response.status_code == 200, response.text
                        body = response.json()
                        observed = {p["label"]: p["probability"] for p in body["predictions"]}
                        mean = expected[domain][:count].mean(0)
                        mean /= mean.sum()
                        error = float(np.abs(np.array([observed[c] for c in CLASSES]) - mean).max())
                        assert error <= 1e-6, error
                        results.append(
                            {
                                "domain": domain,
                                "image_count": count,
                                "http_status": 200,
                                "max_probability_error": error,
                                "seconds": elapsed,
                            }
                        )
                        print(
                            f"{domain} {count} images: HTTP200, error={error:.3g}, {elapsed:.3f}s",
                            flush=True,
                        )
                for data, code in ((b"", "image_empty"), (b"broken", "image_corrupt")):
                    response = client.post(
                        "/v1/diagnose",
                        headers=headers,
                        files=[("images", ("x.png", data, "image/png"))],
                    )
                    assert response.status_code == 422 and response.json()["error_code"] == code
                stream = io.BytesIO()
                Image.new("RGB", (224, 224), "white").save(stream, format="PNG")
                response = client.post(
                    "/v1/diagnose",
                    headers=headers,
                    files=[("images", ("x.png", stream.getvalue(), "image/png"))],
                )
                assert (
                    response.status_code == 422
                    and response.json()["error_code"] == "foreground_not_found"
                )
                files = [("images", (p.name, p.read_bytes(), "image/jpeg")) for p in sources["raw"]]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    pending = [
                        pool.submit(client.post, "/v1/diagnose", headers=headers, files=files)
                        for _ in range(2)
                    ]
                    parallel = sorted(p.result().status_code for p in pending)
                assert parallel == [200, 503], parallel
                assert client.get("/v1/ready").status_code == 200
                schema = client.get("/openapi.json", headers=headers)
                assert schema.status_code == 200
                (args.output / "openapi.json").write_text(
                    json.dumps(schema.json(), indent=2), encoding="utf-8"
                )
                report = {
                    "scope": "five validation samples; technical parity only, not disease accuracy",
                    "model": info,
                    "selected_ids": [r["id"] for r in rows],
                    "requests": results,
                    "parallel_upload_statuses": parallel,
                    "input_and_auth_checks_passed": True,
                    "target": "external_container" if args.base_url else "local_uvicorn",
                    "reference_runtime": {"torch": torch.__version__, "ultralytics": "8.4.72"},
                }
                (args.output / "report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


def main():
    parser = argparse.ArgumentParser(description="Check CVIO native/API probability parity")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url")
    check(parser.parse_args())


if __name__ == "__main__":
    main()
