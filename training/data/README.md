# Dataset card và manifest

## Dataset có group thật và fixture legacy

Không có ảnh tôm thật hoặc weights được phân phối trong repo. Với dataset mới
khác CVIO, cần ghi quyền dùng ảnh, nguồn xác minh nhãn và quy trình PCR nếu dùng
làm ground truth WSSV. Quyền dùng CVIO đã xác nhận được ghi trong card bên dưới.

```text
training/data/shrimp-v1/
  manifest.json
  images/sample-001.jpg
  images/...
```

Manifest JSON gồm `dataset_version`, `split_id`, `synthetic` (boolean), `root`
(tương đối với manifest), `samples` (array). Mỗi sample bắt buộc có:

```json
{"id":"sample-001","group_id":"farm-a-batch-012","label":"BG","split":"train","path":"images/sample-001.jpg","sha256":"<SHA256 thực của file>"}
```

Đây là schema minh họa, không phải dataset chạy được. Tất cả 4 nhãn phải xuất hiện
trong mỗi split train/val/test. Chia theo group trước khi augment; chọn group theo
đơn vị độc lập lớn nhất (cá thể/lô; nếu cùng lô có nhiều cá thể thì dùng lô).
Ghi quyết định grouping trong dataset card, không tạo group ID ngẫu nhiên cho từng
ảnh thật. Ảnh phải >=224×224. Kiểm tra group overlap, ID/path trùng, file hash và
ảnh giống hệt giữa split sẽ chạy trước training/evaluate. Near-duplicate khác
encoding và sai group metadata vẫn cần người kiểm duyệt dataset.

Tạo dữ liệu giả bằng `python scripts/make_toy_dataset.py`; lệnh từ chối ghi đè.
Sau đó dùng config `training/configs/toy.yaml`. Synthetic được giữ trong artifact
và eval; service mặc định từ chối artifact synthetic.

Dataset mới có group thật có thể khai báo `split_policy=group_safe`,
`split_unit=group`, `group_identity=biological`. Validator mặc định vẫn kiểm group
không giao split; nó không tự xác minh sự thật sinh học của metadata. Fixture
synthetic không đại diện cá thể tôm thật. Policy CVIO riêng được mô tả dưới đây.

## CVIO SDI-4 — Phase 1

Nguồn: [Kaggle uynnhy/processed-images](https://www.kaggle.com/datasets/uynnhy/processed-images), **version 1**, manifest version nội bộ **`cvio-sdi4-kaggle-v1`**. Kaggle ghi cập nhật 09/02/2026, license metadata **Unknown**. Chủ đồ án xác nhận nhóm CVIO cho phép dùng mã nguồn, weights và dataset. Không diễn giải Unknown thành public domain hoặc tự suy ra quyền tái cấp phép. [Metadata đã lưu ngày 15/10/2026](../references/kaggle_metadata_v1.json).

Ảnh/weights không phân phối trong Git repo. Dataset import local ở `training/data/cvio-sdi4-kaggle-v1/` được `.gitignore` loại trừ. Adapter không download, không train và không được API gọi lúc startup.

### Reference và counts

Membership chuẩn: `artifacts/manifests/split_manifest_seed42.csv` từ [CVIO commit fd64876](https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26/blob/fd64876cd2fd3acb24640b729d4f019996ebb39b/artifacts/manifests/split_manifest_seed42.csv), branch `report/final-xelatex-e2e-2026-08-11`. Bản nguyên byte ở [reference CSV](../references/cvio_split_seed42.csv); nguồn/license ở [NOTICE](../cvio_core/NOTICE).

SHA-256 reference:

```text
c02e7f0b9a94cb648bca11decfbaa576503c297de849f6e027599a8ed323a965
```

| Lớp | Tổng | Train | Val | Test |
|---|---:|---:|---:|---:|
| Healthy | 403 | 282 | 60 | 61 |
| BG | 198 | 139 | 30 | 29 |
| WSSV | 328 | 229 | 49 | 50 |
| WSSV_BG | 220 | 154 | 33 | 33 |
| Tổng | **1.149** | **804** | **172** | **173** |

Adapter dùng **lớp/tên file** trong reference, kiểm bijection toàn bộ ảnh. Stratified 70/15/15 seed 42 chỉ dùng đối chiếu; không sinh lại split chính khi có reference, không thay membership để đạt metric mong muốn.

### Bytes, hình ảnh và giới hạn nhãn

Adapter copy nguyên luồng byte, không resize/re-encode/composite alpha hoặc xóa nền lại. SHA-256 từng ảnh tính **ở lần import hiện tại** và kiểm lại trước công bố manifest. Cả 1.149 ô MD5 của reference trống; hash mới không chứng minh byte ảnh lần train gốc. Metadata ghi `historical_training_bytes_verified=false`.

Audit xác nhận 1.149 ảnh **RGBA có alpha không hoàn toàn opaque**, thuộc miền ảnh đã tách nền; tất cả đủ cạnh tối thiểu 224. Không phát hiện exact byte/RGB-pixel duplicate trong bản tải đã audit. Adapter vẫn kiểm exact byte duplicate mỗi lần chạy, kể cả cùng split. Near-duplicate và độc lập sinh học chưa được chứng minh; module chưa tự xác minh nhãn bằng chuyên gia/PCR.

### Image-level / unknown group

Manifest khai báo `dataset_version=cvio-sdi4-kaggle-v1`, `split_policy=reference_image_level`, **`split_unit=image`**, **`group_identity=unknown`**, `reference_manifest_sha256`, `synthetic=false`, `root=.`. Sample có `id`, `label`, `split`, `path`, `sha256`, **`group_id: null`**; không lấy sample ID giả làm group sinh học.

Audit nhóm tên file thấy 242/416 prefix giao nhiều split. Prefix chưa được xác nhận là cá thể, nhưng dữ liệu hiện không đủ để tuyên bố group-safe. Validate phải opt-in tường minh:

```python
from training.dataset import REFERENCE_IMAGE_LEVEL, validate_manifest
manifest, root, counts = validate_manifest(path, policy=REFERENCE_IMAGE_LEVEL)
```

`validate_manifest(path)` vẫn mặc định group-safe và từ chối CVIO. Policy compatibility vẫn kiểm ID/path/hash/duplicate và đủ bốn lớp trong mỗi split. Runner CVIO sẽ opt-in policy này tường minh và kiểm lại membership với reference trước train/evaluate; pipeline tương thích artifact v1 không được tự nhận CVIO là group-safe.

### Quality gate — phát hiện dành cho Phase 3

Kết quả audit ngày 15/10/2026 với **blur threshold 50**:

| Split | Chấp nhận | Từ chối vì blur | Tổng |
|---|---:|---:|---:|
| Train | 63 | 741 | 804 |
| Val | 16 | 156 | 172 |
| Test | **9** | **164** | **173** |
| Tổng | **88** | **1.061** | **1.149** |

Đây là kết quả heuristic, không kết luận 1.061 ảnh vô dụng. **Phase 1 không đổi gate/ngưỡng blur** và không loại ảnh khỏi split theo gate. Phase 3 phải đánh giá lại bằng validation và ảnh sản phẩm phù hợp; tách full-test metric với coverage/metric trên ảnh được nhận. Miền RGBA đã xử lý khác ảnh chụp tại ao còn nền.

### Chạy adapter

Từ root repo:

```powershell
.\.venv\Scripts\python.exe -m training.cvio_data --source ../.tmp/cvio-processed-images-v1.zip --output training/data/cvio-sdi4-kaggle-v1
```

Cũng chấp nhận directory giải nén. Inventory dùng directory lớp trực tiếp chứa ảnh; bỏ qua file không phải ảnh, từ chối ảnh thừa/thiếu/key trùng và symlink/path không hợp lệ. CLI production luôn pin reference/counts 1.149, không có flag bỏ kiểm tra.

Output gồm `images/`, `manifest.json`, `import_report.json`. Directory đích phải chưa tồn tại. Nếu copy/import lỗi giữa chừng, directory có thể chứa ảnh nhưng **không có manifest.json hoàn tất**; xem lỗi và dùng directory mới, không tự ghi đè/xóa source. Source ZIP có archive SHA được ghi trong manifest.

`python -m training.cvio_data --compare-stratified` là diagnostic tùy chọn, cần `scikit-learn` trong môi trường thực nghiệm. Base service không thêm dependency này. Chế độ đó chỉ in membership khác biệt, không nhận output và không ghi split. Ghi phiên bản sklearn trong log nếu sử dụng; kết quả không thay reference chuẩn.
