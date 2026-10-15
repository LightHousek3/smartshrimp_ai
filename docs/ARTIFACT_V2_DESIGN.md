# Thiết kế artifact v2

Schema: training/schemas/artifact_v2.schema.json. Đây là thiết kế metadata;
loader/exporter v2 chưa được triển khai ở mốc dữ liệu.

Các trường: schema_version=2, backend=onnxruntime, output_kind=probabilities,
class_order cố định, preprocessing/gate snapshot, source_checkpoint_sha256,
onnx_sha256, dataset/split/manifest/config identity và synthetic flag.

Ultralytics classification export trả probabilities: không softmax hai lần.
Preprocessing v8.4.72 là RGB, resize cạnh ngắn + center crop 224, mean 0/std 1.
Không dùng stretch + ImageNet normalization của artifact v1 cho model mới.

Chọn best theo fitness Ultralytics, không đổi sang macro-F1. Macro-F1 checkpoint
selection phải được đặt tên như biến thể khác. Giữ quality gate blur=50 ở giai
đoạn hiện tại; ngưỡng cần được đánh giá bằng validation trước triển khai.

V2 cần hash verification, provenance và parity cả vector xác suất với batch
1/2/5, ảnh không vuông, alpha và EXIF. Thiết kế này chưa chứng minh các kiểm tra
đó đã chạy. API v1 tương thích được bổ sung ở milestone riêng.
