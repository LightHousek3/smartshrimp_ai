# Hợp đồng SmartShrimp backend ↔ AI service

Service stateless: không kết nối PostgreSQL, không lưu ảnh/kết quả lâu dài.
Multipart có thể spool tạm ở `/tmp` và được đóng sau request. Backend quản lý
object storage, URL ảnh, lịch sử, retention và phân quyền. Không gửi API key xuống
Flutter/web; chỉ backend gọi AI qua mạng nội bộ/TLS.

Đối chiếu schema `smartshrimpv2.sql`, bảng `ai_diagnosis_logs` (dòng 506):

| Field | Backend ghi |
|---|---|
| id, created_at | DB sinh theo schema hiện tại |
| season_id, requested_by | Backend xác thực KTV được phân công vào vụ ACTIVE |
| image_urls | TEXT[] URL ảnh đã lưu, cùng thứ tự multipart images, 1–5 ảnh cùng ca quan sát |
| input_snapshot | JSONB chứa request_id, model_version, checkpoint_sha256, preprocessing, quality_gate, image_count và thông tin đầu vào nghiệp vụ |
| run_status | `success` khi HTTP 200, `error` khi lỗi HTTP/timeout/network |
| predicted_label | Một trong Healthy/BG/WSSV/WSSV_BG, NULL khi error |
| confidence | Float [0,1] → NUMERIC(6,5), chỉ làm tròn khi lưu cột này |
| predictions | JSONB đủ 4 label/probability, giữ precision gốc, giảm dần; [] khi error |
| recommendation | Chuỗi tiếng Việt API trả, NULL khi error |
| model_version | Version API trả; với lỗi trước inference dùng metadata đã cache nếu biết, không đoán |
| processing_time_ms | API trả trên success; lỗi kỹ thuật dùng duration backend đo, ghi nguồn đo trong snapshot |
| error_code, error_message | NULL khi success; mã máy đọc và thông báo ngắn khi error |

`confidence` trong DB có thể lệch cực nhỏ với top probability do NUMERIC(6,5);
không làm tròn riêng từng probability để tránh tổng khác 1.

Backend gọi `/v1/model/info` khi triển khai model và cache theo model_version +
checkpoint_sha256. Endpoint này cung cấp preprocessing/quality_gate để đưa vào
snapshot. Version artifact bất biến; thay preprocessing hay weights phải tạo
version mới. Khi rolling deployment, cache metadata theo version đáp ứng, không
gắn metadata version A vào kết quả version B. Giữ một instance trong scope hiện tại.

API không nhận season_id/account_id, không tự mở Disease Case, không tạo phác đồ,
không yêu cầu Expert đánh giá AI. Recommendation là tham khảo theo scope đồ án.

## Lỗi

JSON: `{"error_code":"image_too_blurry","error_message":"...","request_id":"..."}`.
Mọi response có `X-Request-Id`. Backend sinh request ID nếu cần correlation, gửi
header; service chỉ chấp nhận 1–64 ký tự ASCII chữ/số/dấu chấm/gạch ngang/gạch dưới.

| HTTP | error_code |
|---|---|
| 401 | unauthorized |
| 422 | invalid_image_count, invalid_multipart, invalid_request, image_empty, image_corrupt, image_too_large, request_too_large, unsupported_image_format, image_too_small, image_too_many_pixels, animated_image_not_supported, image_too_blurry |
| 429 | rate_limit_exceeded (kèm Retry-After) |
| 503 | model_not_ready |
| 500 | model_inference_failed, internal_error |

Backend tự đặt `ai_timeout`/`ai_unavailable` cho lỗi transport. Không biến lỗi thành
Healthy hoặc confidence=0. Không retry ảnh 422/auth 401. Timeout có thể xảy ra sau
khi model đã xử lý; backend chịu trách nhiệm chống ghi trùng lịch sử theo request ID.

1–5 ảnh phải thuộc cùng đối tượng/ca quan sát; service lấy trung bình softmax từng
ảnh, rồi chuẩn hóa và chọn top. Không gửi ảnh các ao khác nhau trong một request.
Nếu bất kỳ ảnh nào không đạt gate, toàn request bị từ chối. Giới hạn là 10 MiB/file,
tổng multipart <=50 MiB +64 KiB overhead; decoded image mặc định <=20 megapixel.

Rate limit sliding window 60 giây/key áp dụng diagnose + model/info sau auth;
request không hợp lệ vẫn tiêu quota. `/metrics` có auth, không tiêu quota để scrape.
Health/ready/live công khai trong mạng nội bộ. Triển khai mặc định đúng 1 worker,
1 replica: limiter in-memory không phải quota toàn cluster. Nếu scale, đưa quota
về gateway dùng chung; không thêm database nghiệp vụ vào AI service.
