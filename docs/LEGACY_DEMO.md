# Artifact v1 — compatibility và demo

Mô hình nộp là YOLO26m-cls + recipe CVIO. Layer v1 giữ khả năng đọc artifacts
của service trước và chạy API bằng synthetic tiny fixture. Các factory legacy
không được coi là backbone nghiên cứu chính; không cung cấp experiment config
EfficientNet/CSP như phương án nộp thay CVIO.

```bash
python scripts/make_toy_dataset.py
python -m training.train --config training/configs/toy.yaml
python -m training.evaluate --checkpoint models/shrimpdx-toy-v1.0.0/weights.pt --split test
python -m training.export --checkpoint models/shrimpdx-toy-v1.0.0/weights.pt --format onnx
```

Chạy từ root với requirements-dev.txt. CLI từ chối ghi đè version. Synthetic
không có giá trị chẩn đoán; production mặc định từ chối artifact synthetic.
Để demo local, tạo .env với API_KEY ngẫu nhiên ít nhất 24 ký tự, rồi đặt
MODEL_DIR=models/shrimpdx-toy-v1.0.0 và ALLOW_SYNTHETIC_MODEL=true.

```bash
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Docker demo dùng `docker compose -f docker-compose.yml -f docker-compose.demo.yml up --build`.
Docker/CI remote chưa được xác minh ở phiên local này; không coi có YAML là đã
nghiệm thu container. API nhận 1–5 JPEG/PNG/WebP, >=224×224, <=10 MiB/file.
X-API-Key bắt buộc; rate limit 60/phút/key; blur threshold 50 không đổi.

Artifact v1 chứa state_dict và config/metadata/manifest/eval có hash. Loader
kiểm provenance, checksum, ontology và probabilities trước readiness. V1 dùng
stretch + ImageNet normalization và mean softmax; contract này chỉ dành v1.
CVIO dùng resize cạnh ngắn + center crop 224, mean 0/std 1, export probabilities.
Migration loader/ONNX của CVIO thuộc Phase 3, không thay bytes của artifact v1.

Nội dung v1 được giữ từ implementation service trước để regression tương thích;
CVIO adapter/core/runner không phụ thuộc factory v1 để train mô hình chính.
