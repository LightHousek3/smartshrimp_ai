# Protocol đánh giá và công bố

Với CVIO: split image-level, group identity unknown; không tuyên bố group-safe.
Chọn checkpoint theo fitness Ultralytics; macro-F1 là metric báo cáo.
Các đường dẫn report v1 dưới đây thuộc pipeline compatibility; runner CVIO
lưu predictions/metrics riêng trong thư mục run. Test cuối sau khi đóng băng
model/preprocessing/gate, không dùng test để chọn lại recipe.


1. Dataset card ghi nguồn, quyền dùng, quy trình xác nhận nhãn, thiết bị, điều kiện
   chụp, group đơn vị cá thể/lô và giới hạn dữ liệu.
2. Đóng băng version, split ID, seed và manifest có SHA256 từng file. Audit group
   overlap và duplicate bytes tự động; audit near-duplicate và cùng ao/lô thủ công.
3. Train chỉ train; lựa chọn epoch/hyperparameter chỉ val. Ghi cấu hình, package
   versions, phần cứng, seed, checkpoint hash trong artifact.
4. Sau khi chọn model mới evaluate test. Không lựa chọn lại model theo test.
   Nếu làm nhiều experiment, phải công bố quy trình chọn và tránh test leakage.
5. Chỉ trích metric từ `reports/eval_<model_version>.json`. `history.json` chỉ
   phục vụ debug train; `eval.json` trong artifact là bản sao report test.

`accuracy` = số đúng / tổng; precision_i=TP_i/(TP_i+FP_i), recall_i=TP_i/support_i,
F1_i=2PR/(P+R); denominator=0 cho 0; macro-F1 là trung bình **đủ 4** F1.
Confusion matrix: hàng ground truth, cột predicted, thứ tự ontology trong report.
Report chứa per-class precision/recall/F1/support, sample_count, evaluated_split,
dataset_version, split_id, seed, manifest/config/checkpoint SHA256 và command.

`publishable: true` chỉ có nghĩa là report test từ manifest không synthetic;
không tự chứng nhận nhãn thật, hiệu quả thực địa, tính độc lập của dataset hay
độ tin cậy thống kê. Hội đồng/nhóm nghiên cứu vẫn phải duyệt dataset và protocol.
Toy report luôn `synthetic: true`, `publishable: false`.

Trước pilot thực địa: đo recall từng bệnh, false-negative, robustness theo ao,
thiết bị, ánh sáng, nền và blur; báo cáo cỡ mẫu và khoảng tin cậy. Confidence là
softmax score chưa calibration, không phải xác suất bệnh được xác nhận lâm sàng.
Model 4 lớp không có lớp OOD/không phải tôm; quality gate chỉ kiểm tra file và
chất lượng ảnh. Không coi ảnh bất kỳ vượt gate là ảnh tôm đã được xác minh.
