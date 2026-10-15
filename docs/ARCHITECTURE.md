# Kiến trúc và phạm vi

SmartShrimp sử dụng classification bốn lớp theo thứ tự Healthy, BG, WSSV,
WSSV_BG. Core mô hình nộp là YOLO26m-cls + recipe CVIO. CE và proposed dùng cùng
reference split, seed và hyperparameters; không bổ sung một backbone nghiên cứu khác.

## Các mốc tích hợp

1. Foundation: hợp đồng dữ liệu/API, nguồn, giấy phép và recipe.
2. Data: adapter membership CVIO, checksum, dataset card và artifact v2 design.
3. Training core: port loss/attention, runtime pin, pretrained và reload/resume tests.
4. API compatibility: giữ dịch vụ artifact v1 để kiểm hợp đồng/demo trong khi
   checkpoint CVIO và artifact v2 còn chờ nghiệm thu. Đây là mã tương thích kế
   thừa phiên bản service trước, không phải mô hình nộp thay YOLO26m-cls.
5. Runner: thực nghiệm CE/proposed, Kaggle và evidence package.

Sau thực nghiệm: export/ONNX/preprocessing/gate ở Phase 3 và nghiệm thu hệ thống
ở Phase 4. Chưa có checkpoint GPU đầy đủ không đồng nghĩa có model production.

## Ranh giới hệ thống

Backend gọi service qua API nội bộ và giữ secret; backend lưu ảnh/lịch sử theo
vụ nuôi. Service trả nhãn, confidence, bốn probabilities, recommendation,
model_version và thời gian xử lý. Không tự tạo Disease Case, phác đồ hoặc RAG.

Ảnh CVIO đã tách nền RGBA; split mức ảnh và group identity unknown. Chỉ dùng
reference membership, không tái chia theo seed. Preprocessing model mới phải
theo runtime Ultralytics; blur gate mặc định 50 chờ đánh giá Phase 3.

Các commit được tạo tại thời điểm tích hợp lại repository. Giữ nguồn và license
CVIO; các kiểm thử/qualification có phạm vi riêng với nghiệm thu GPU và thực địa.
