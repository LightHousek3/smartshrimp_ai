# CVIO training core

Source/license ở training/cvio_core/NOTICE. Port ASL-LDAM và SimAM-DCFR theo
commit CVIO fd64876. Golden fixtures được tính trực tiếp từ source đã pin,
không tính expected values từ implementation đang kiểm tra.

Tích hợp Ultralytics 8.4.72: chuyển toàn bộ loss config, lấy logits từ phần tử
thứ hai của tuple eval, nạp pretrained trước attention và suy luận head channels
bằng probe. Resume giữ full custom model và kiểm optimizer/EMA state.

Recipe chính: YOLO26m-cls pretrained, 224, 30 epoch, batch 32, workers 4, AMP,
patience 15, AdamW, lr0=.00125, lrf=.01, cos_lr, randaugment, cache=false, seed=42.
Train counts LDAM: 282/139/229/154. Best theo Ultralytics fitness.

```bash
python -m pip install -r requirements-training.txt
python -m pytest tests/test_cvio_core.py -q
```

Các test structural offline dùng kiến trúc nhỏ để kiểm module/gradient/reload;
qualification với pretrained model m và thực nghiệm GPU cần được ghi riêng.
CPU autocast không chứng minh AMP CUDA. Runner dài và evidence package thuộc
milestone tiếp theo; không có metrics GPU hoặc model production ở mốc core.
