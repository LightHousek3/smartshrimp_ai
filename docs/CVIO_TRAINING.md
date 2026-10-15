# Huấn luyện CVIO SDI-4 cho SmartShrimp

Tài liệu vận hành Phase 2, cập nhật 15/10/2026. Mã đã được tách IO dùng chung
để adapter/core CVIO độc lập với loader compatibility v1. Mục tiêu là hai thực nghiệm
YOLO26m-cls: CE và ASL-LDAM + SimAM-DCFR. Phần tích hợp đã được kiểm tra local
CPU; hai thực nghiệm GPU đầy đủ **chưa chạy**. Không dùng số smoke làm kết quả
nghiên cứu. API hiện vẫn dùng artifact v1; ONNX và chính sách ảnh thuộc Phase 3.

## 1. Nguồn và phạm vi

- [CVIO tại commit fd64876](https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26/tree/fd64876cd2fd3acb24640b729d4f019996ebb39b),
  branch `report/final-xelatex-e2e-2026-08-11`.
- Core kế thừa mang AGPL-3.0: xem [NOTICE](../training/cvio_core/NOTICE) và
  [toàn văn giấy phép](../training/cvio_core/LICENSE). LICENSE MIT của service
  ban đầu không cấp lại giấy phép cho phần CVIO/Ultralytics tích hợp.
- Dataset `uynnhy/processed-images`, **version 1**, metadata license Unknown;
  chủ dự án xác nhận nhóm CVIO cho phép sử dụng source/weights/dataset.
  Xem [dataset card](../training/data/README.md). Không phân phối ảnh/weights
  trong gói source.
- Reference CSV quyết định membership theo lớp/tên file; stratified seed 42
  chỉ để đối chiếu. `split_unit=image`, `group_identity=unknown` vì chưa có
  danh tính cá thể/phiên chụp. Không tuyên bố độc lập sinh học.

## 2. Recipe và các sửa đổi tích hợp

| Thuộc tính | Giá trị |
|---|---|
| Backbone / khởi tạo | YOLO26m-cls / pretrained chính thức 1.000 lớp, thay head 4 lớp |
| Class order | Healthy, BG, WSSV, WSSV_BG |
| Input / epoch / batch / workers | 224 / tối đa 30 / 32 / 4 |
| AMP / patience / optimizer | true / 15 / AdamW |
| lr0 / lrf / cos_lr | 0.00125 / 0.01 / true |
| Augment / cache / seed | randaugment / false / 42 |
| ASL | gamma_pos=0, gamma_neg=4, label_smoothing=0.1 |
| LDAM | counts train=[282,139,229,154], max_m=0.5, scale=30 |
| Attention | SimAM + gated residual + DCFR texture, epsilon=0.0001 |
| Best checkpoint | fitness Ultralytics = (top1 + top5)/2 |

Với bốn lớp, top5 luôn bao phủ toàn bộ lớp. Không đổi checkpoint selection sang
macro-F1; một thực nghiệm như vậy phải mang tên biến thể riêng. Khi fitness hòa,
giữ hành vi save best nguyên bản của Ultralytics. Macro-F1 là metric báo cáo.

Port giữ nguyên công thức loss/attention. Tích hợp sửa lỗi tuple eval lấy
probabilities thay logits, cấu hình loss bị bỏ qua, run_name/path không nhất
quán, status/skip sai và resume có thể làm mất custom head. Pretrained được nạp
và so từng tensor tương thích **trước** khi thêm attention. Head nhận feature
512×7×7 trên model m; attention thêm 529.408 tham số.

Trainer lưu optimizer/scaler trong best/last; không strip tại final_eval.
Runner đánh giá best trên validation, không tự đánh giá test sau train.
Resume khôi phục custom module, EMA và optimizer; chưa cam kết chuỗi RNG/
dataloader hoặc kết quả bitwise giống một run không gián đoạn.

Runtime qualification: Ultralytics **8.4.72**, torch **2.9.1**, torchvision
**0.24.1**, Python **3.12**. Đây là môi trường tích hợp được khóa, không phải
tuyên bố tìm lại chính xác môi trường official-final của CVIO.

Preprocessing validation lưu cùng checkpoint: resize cạnh ngắn + center crop
224, RGB, mean=0/std=1. Ảnh RGBA được đọc RGB theo ClassificationDataset của
runtime khóa phiên bản; không tự thêm composite/rembg. Khi export ở Phase 3,
Ultralytics trả **probabilities**; engine không được softmax thêm lần nữa.
Preprocessing stretch + ImageNet normalization của service v1 không dùng cho
model mới. Quality gate và blur=50 chưa đổi ở phase này.

## 3. Chuẩn bị gói source và Kaggle

Tại repo local, dùng Python của `.venv`:

```powershell
.\.venv\Scripts\python.exe -m scripts.package_cvio_source --output .tmp/delivery/cvio-phase2-source.zip
```

ZIP chứa source, tests, config, reference CSV, tài liệu vận hành và notices theo
allowlist; có `SOURCE_SHA256SUMS`. Không chứa `.git`, `.env`, cache, dữ liệu hoặc
checkpoint. Không ghi đè ZIP đã tồn tại.
Đây là snapshot source đang có trên đĩa, kể cả thay đổi chưa commit; không dựa
vào một git clone còn thiếu mã mới. Không sửa/xóa lịch sử nguồn ở repo local.

Tạo notebook Kaggle, bật Internet lúc cài dependencies/tải pretrained, chọn
GPU. Attach dataset version 1 và source ZIP dưới dạng input riêng. Chép/giải
nén source vào `/kaggle/working/smartshrimp_ai`; thư mục `/kaggle/input` chỉ đọc.
Ví dụ cell Python (thay đường dẫn ZIP bằng input thực tế):

```python
from pathlib import Path
from zipfile import ZipFile
import hashlib
import subprocess
import sys

source_zip = Path('/kaggle/input/YOUR-SOURCE/cvio-phase2-source.zip')
repo = Path('/kaggle/working/smartshrimp_ai')
repo.mkdir(exist_ok=False)  # chạy một lần; không ghi đè một checkout đang dùng
with ZipFile(source_zip) as archive:
    for name in archive.namelist():
        if not (repo / name).resolve().is_relative_to(repo.resolve()):
            raise ValueError('Unsafe ZIP path')
    archive.extractall(repo)
for line in (repo / 'SOURCE_SHA256SUMS').read_text().splitlines():
    digest, name = line.split('  ', 1)
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
```

Dùng venv riêng để không xung đột các package có sẵn trong notebook:

```python
assert sys.version_info[:2] == (3, 12), 'Qualification này yêu cầu Python 3.12'
venv = Path('/kaggle/working/cvio-venv')
subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
python = str(venv / 'bin/python')
def run(*args):
    subprocess.run([python, *args], cwd=repo, check=True)

run('-m', 'pip', 'install', 'torch==2.9.1', 'torchvision==0.24.1',
    '--index-url', 'https://download.pytorch.org/whl/cu128')
run('-m', 'pip', 'install', '-r', 'requirements-training.txt')
run('-m', 'pip', 'check')
run('-m', 'pytest', '-q')
run('-m', 'ruff', 'check', '.')
run('-c', 'import torch; print(torch.__version__, torch.version.cuda); '
    'assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))')
```

Cặp CUDA wheel theo [hướng dẫn PyTorch chính thức](https://pytorch.org/get-started/previous-versions/).
Driver/CUDA, package Linux và AMP phải qua qualification trên phiên thật;
không suy từ kiểm tra CPU. Nếu Kaggle không có Python 3.12 hoặc CUDA wheel
không tương thích driver, dừng và lưu lỗi để điều chỉnh môi trường có kiểm thử,
không tự đổi hàng loạt version/recipe. Chỉ dùng GPU `0`; không sử dụng DDP T4×2.
Các callback tracking bên thứ ba được tắt.

## 4. Import và kiểm thử GPU ngắn

```python
runner = ['-m', 'scripts.run_cvio_kaggle']
run(*runner, 'prepare', '--dataset', '/kaggle/input/processed-images')
run(*runner, 'smoke', '--method', 'ce')
run(*runner, 'smoke', '--method', 'proposed')
```

`prepare` kiểm bijection 1.149 ảnh, checksum, duplicate và membership, lưu bản
ảnh nguyên byte dưới `cvio-work/data`. Pretrained tải từ release chính thức
v8.4.0, SHA-256 phải là
`9f6546f33a70d910e2cd6dcca5265c4617b5670b19a1f287c39e99258bade01a`.
Source ZIP không tự chứa pretrained.

Smoke dùng 8 train/4 validation, 2 epoch, batch=2, workers=0; trên GPU giữ AMP.
Đây là ngoại lệ được ghi `smoke=true`, không phải run recipe chính.
Kiểm `effective_runtime.json` xác nhận AMP bật. Stock Ultralytics có thể thử
tải model nhỏ cho AMP check; log ghi rõ nếu check bị skip do offline.
Smoke custom model vẫn phải chạy thành công trước run dài.

## 5. Hai run chính và cách tiếp tục

```python
run(*runner, 'train', '--method', 'ce')
run(*runner, 'train', '--method', 'proposed')
```

Tên cố định `ce-seed42` và `proposed-seed42`; cùng manifest, seed và recipe.
Runner chỉ stage train/val, không tạo folder test trong cây Ultralytics.
`resolved_args.json` ghi cả defaults của runtime. Nếu OOM, lưu lỗi; không đổi
batch rồi gọi đó là cùng run chuẩn. Không hứa thời gian hoàn thành hoặc accuracy.

Nếu gián đoạn sau khi có last.pt:

```python
run(*runner, 'train', '--method', 'proposed', '--resume')
# Chỉ bỏ qua một run đã complete và có checksum hợp lệ:
run(*runner, 'train', '--method', 'ce', '--skip-complete')
```

Resume yêu cầu đúng manifest, recipe, pretrained và loại smoke/full. Không
fallback khởi tạo lại nếu mất checkpoint. Khôi phục cả thư mục run và **manifest
gốc** khi chuyển phiên; import mới có timestamp khác sẽ làm đổi hash manifest.
Đặt ảnh dưới root tương đối của manifest gốc, không chỉnh manifest để qua guard.
Giữ logs và environment của các phiên. `last.pt` có optimizer/EMA; `best.pt`
phục vụ đánh giá. Chỉ nạp checkpoint do run của dự án tạo hoặc nguồn chính thức
đã kiểm hash; PyTorch checkpoint chứa cấu trúc Python.

Run identity còn khóa SHA-256 source training và version packages; không sửa
code giữa một run rồi resume âm thầm. Nếu đã train xong nhưng bị ngắt lúc xuất
validation, `--resume` kiểm `training_completed.json` và tiếp tục finalization,
không chạy thêm epoch. Evaluation ghi ra thư mục tạm rồi đổi tên khi đủ file;
các thư mục `.pending-*` không được coi là bằng chứng hoàn tất.

## 6. Bằng chứng và đóng băng trước test

Mỗi run thành công tạo ZIP trong `/kaggle/working/cvio-work/evidence`, gồm
best/last, SHA-256, history `results.csv`, resolved args, environment + pip freeze,
manifest, class order/module audit, validation predictions đủ bốn xác suất,
accuracy/macro-F1/per-class/confusion/kappa và status. ZIP không chứa ảnh stage.
Tải ZIP về trước khi kết thúc phiên. Lưu thêm notebook output và `cvio-work/logs`
khi thất bại; notebook session có thể bị mất nếu chưa lưu output.

Đóng gói lại có tên mới:

```python
run(*runner, 'package', '--method', 'ce')
run(*runner, 'package', '--method', 'proposed')
```

Hai ZIP Phase 2 chỉ có validation: chưa hoàn tất nghiệm thu thực nghiệm trên
test. Ở Phase 3 dùng validation để kiểm export/parity và chính sách ảnh; sau khi
đóng băng checkpoint/recipe/preprocessing/gate, Phase 4 mới chạy đánh giá test
cuối. CLI đã có để dùng đúng thời điểm:

```python
work = Path('/kaggle/working/cvio-work')
for method in ('ce', 'proposed'):
    run('-m', 'training.cvio_run', 'evaluate', '--run', str(work/'runs'/f'{method}-seed42'),
        '--manifest', str(work/'data/manifest.json'), '--split', 'test')
    run(*runner, 'package', '--method', method)
```

Không ghi đè evaluation có sẵn và không cho smoke đọc test. Metric test từ
checkpoint này mới là kết quả dự án; số published của CVIO chỉ là đối chiếu.
Không dùng test để chọn lại hyperparameter mà vẫn gọi test độc lập. Đánh giá
classifier này chưa áp quality gate API; coverage/conditional performance và
ảnh thực địa cần báo cáo riêng. Gate hiện từ chối 164/173 ảnh test ở blur=50.

## 7. Checklist bàn giao từ GPU

- [ ] Hai smoke GPU thành công, effective AMP=true, module audit đúng.
- [ ] CE và proposed hoàn thành 30 epoch hoặc early stop có history xác minh.
- [ ] Best/last tải được; hashes/manifest/class order/environment đầy đủ.
- [ ] Full validation predictions có 172 hàng mỗi run, probabilities hợp lệ.
- [ ] Hai evidence ZIP và logs được tải/lưu ngoài phiên notebook.
- [ ] Kiểm tra artifact/export/API ở Phase 3, sau đó đánh giá test cuối Phase 4.

Không chuyển API sang model mới chỉ vì có checkpoint. Loader v1, bốn nhãn và
ngưỡng gate hiện tại vẫn giữ nguyên cho tới phase được duyệt.
