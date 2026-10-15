# SmartShrimp AI service

Module hỗ trợ phân loại ảnh tôm thành `Healthy`, `BG`, `WSSV`, `WSSV_BG`.
Mô hình chính là **YOLO26m-cls kế thừa recipe CVIO**; CE là baseline đối chiếu,
ASL-LDAM + SimAM-DCFR là phương án proposed. Backend quản lý xác thực nghiệp vụ,
lưu ảnh và lịch sử; service không sở hữu database hoặc tự tạo Disease Case.

## Trạng thái

Adapter dữ liệu/reference policy và thiết kế artifact v2 đã có. Chưa port loss/attention hoặc chạy training.

Không sử dụng metrics đã công bố của CVIO như kết quả thực nghiệm SmartShrimp.
Các commit phản ánh các mốc tích hợp source và kiểm thử; chúng không chứng minh
hiệu quả phân loại trên dữ liệu thực địa hoặc một lần train GPU đã hoàn tất.

## Kiểm tra local

Python 3.12 cho môi trường qualification training. Tạo venv riêng rồi chạy:

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
```

Tài liệu: [Kiến trúc và phạm vi](docs/ARCHITECTURE.md) · [Dataset card](training/data/README.md) · [Artifact v2](docs/ARTIFACT_V2_DESIGN.md).

## Nguồn và giấy phép

[CVIO](https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26)
branch `report/final-xelatex-e2e-2026-08-11`, commit `fd64876cd2fd3acb24640b729d4f019996ebb39b`.
Phần CVIO kế thừa mang AGPL-3.0; xem [NOTICE](training/cvio_core/NOTICE) và
[giấy phép nguồn](training/cvio_core/LICENSE). [LICENSE](LICENSE) MIT của phần
service không tái cấp phép cho core CVIO, Ultralytics, weights hoặc dữ liệu.
Quyền sử dụng source/weights/dataset do chủ đồ án xác nhận với nhóm CVIO.
Kaggle ghi license metadata Unknown; không phân phối images/weights trong Git.
