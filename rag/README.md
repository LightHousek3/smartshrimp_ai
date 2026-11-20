# SmartShrimp RAG Service

## API nạp tài liệu từ admin

Admin quản lý tài liệu qua BE tại `/api/v1/admin/rag/documents` (Bearer token ADMIN ACTIVE). BE cấp upload intent để browser tải file lên Cloudinary rồi xác nhận upload và xếp hàng ingestion. File tối đa 10 MiB, định dạng PDF/DOCX/MD/TXT. Xem [quy ước tài liệu BE](../../smartshrimp_be/docs/rag-document-conventions.md).

RAG chỉ nhận tài liệu managed từ BE qua `POST /v1/documents/{uuid}/ingest` với query `filename`, `title`, `attempt`, header `X-Internal-Key` và body binary `application/octet-stream`; xóa vectors qua `DELETE /v1/documents/{uuid}/vectors`. Attempt, advisory lock, transaction và tombstone bảo vệ index. File tạm được dọn sau xử lý. API legacy `POST /v1/documents` và CLI local đã bị loại bỏ.

Schema vector dùng các migration 001/002 trong `DoAn/migrations/`. Backend dùng `RAG_INGESTION_TIMEOUT_MS` (mặc định 300000 ms). Chat gửi danh sách UUID/attempt READY và active, được lọc trước top-K. Danh sách thiếu, null hoặc rỗng không cho phép truy xuất tài liệu nào; không còn truy xuất legacy. Thay đổi này không xóa vector đã lưu. Chạy test bằng `python -m pytest -q`; đặt `RAG_DB_INTEGRATION=1` chỉ khi đã xác minh DB DEV/test.

Microservice FastAPI cung cấp chức năng hỏi đáp kỹ thuật nuôi tôm bằng
Retrieval-Augmented Generation (RAG). Service đọc tài liệu chuyên môn, chia tài liệu
theo ngữ nghĩa, lưu embedding trong PostgreSQL/pgvector và sử dụng Gemini để tạo câu
trả lời dựa trên các đoạn tài liệu tìm được.

README được chia thành hai phần:

1. [Sử dụng RAG như một service độc lập](#phần-1--sử-dụng-rag-như-một-service-độc-lập)
2. [Tích hợp RAG với hệ thống SmartShrimp](#phần-2--tích-hợp-với-hệ-thống-smartshrimp)

> **Trạng thái tích hợp:** Backend đã có tích hợp chat và quản lý tài liệu. Phần 2 giữ mô tả contract nền; luồng quản lý hiện tại theo tài liệu API liên kết ở trên.

## Tổng quan kiến trúc

```text
Tài liệu PDF/DOCX/MD/TXT
          │
          ▼
Loader → Semantic chunking → Gemini Embedding
                                  │
                                  ▼
                         PostgreSQL + pgvector
                                  │
                                  ▼
Câu hỏi → Query embedding → Top-K chunks → Gemini LLM → Câu trả lời
```

### Công nghệ

| Thành phần | Công nghệ |
|---|---|
| API | FastAPI, Uvicorn |
| Embedding | Google Gemini Embedding, vector 768 chiều |
| Sinh câu trả lời | Google Gemini |
| Vector store | PostgreSQL với extension `pgvector` |
| Database client | `asyncpg` |
| Đọc tài liệu | `pypdf`, `python-docx` |
| Chunking | Semantic chunking |
| Kiểm thử | `pytest`, `pytest-asyncio`, `httpx` |

Embedding 2 dùng chỉ dẫn trực tiếp trong text, không gửi `task_type`:

- Câu hỏi: `task: question answering | query: {câu hỏi}`.
- Chunk cuối: `title: {tên tài liệu} | text: {nội dung}`.
- Semantic unit: nội dung thuần, không thêm title hoặc prefix truy xuất.
- Tên tài liệu ưu tiên title của Admin, rồi tên file gốc.

Model cấu hình và output 768 chiều được giữ nguyên. Vector đã tạo bằng định dạng
cũ cần được re-embed trước khi dùng với query mới; thay đổi code không tự re-embed.

Semantic units được embed theo batch 32, chunks cuối theo batch 16 phần tử,
giữ thứ tự vector tương ứng với input. Với 200 chunks cuối, bước này gọi SDK
13 lần; số request tìm ranh giới semantic được tính riêng. Chỉ ghi index sau khi
tất cả batch hoàn thành; lỗi ở một batch không ghi dữ liệu dở dang.

Ingestion luôn dùng semantic chunking; không còn cấu hình chọn strategy.
Điều chỉnh bằng `MIN_CHUNK_SIZE`, `MAX_CHUNK_SIZE`, `CHUNK_OVERLAP` và
`SEMANTIC_BREAKPOINT_PERCENTILE`.

---

# Phần 1 — Sử dụng RAG như một service độc lập

## 1. Yêu cầu

- Python 3.11 được khuyến nghị.
- PostgreSQL có extension `pgvector`.
- Gemini API key.
- Tài liệu kiến thức ở định dạng `.pdf`, `.docx`, `.md` hoặc `.txt`.

## 2. Cài đặt

Từ thư mục `smartshrimp_rag`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Trên macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Cấu hình môi trường

Tạo file riêng `rag/.env`, không commit. Cập nhật tối thiểu các biến sau:

```env
APP_ENV=development
INTERNAL_API_KEY=replace-with-a-long-random-secret
GEMINI_INGESTION_API_KEY=your-ingestion-api-key
GEMINI_QUERY_API_KEY=your-query-api-key
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2
GEMINI_LLM_MODEL=gemini-3.5-flash-lite
GEMINI_FALLBACK_MODELS=["gemini-3.1-flash-lite"]
DATABASE_URL=postgresql://postgres:password@localhost:5432/smartshrimp
```

`DATABASE_URL` chấp nhận cả tiền tố `postgresql://` và
`postgresql+asyncpg://`.

### Các biến cấu hình

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `APP_ENV` | `development` | Vẫn bắt buộc khai báo rõ khi startup; `production` tắt Swagger. |
| `APP_VERSION` | `1.0.0` | Phiên bản trả về ở health check. |
| `INTERNAL_API_KEY` | rỗng | Secret dùng chung với BE, bảo vệ các endpoint nội bộ. |
| `GEMINI_INGESTION_API_KEY` | rỗng | Key cho semantic units và embedding chunks khi nạp tài liệu. |
| `GEMINI_QUERY_API_KEY` | rỗng | Key cho embedding câu hỏi, đếm token và sinh câu trả lời. |
| `GEMINI_EMBEDDING_MODEL` | `models/gemini-embedding-2` | Model embedding. |
| `GEMINI_LLM_MODEL` | `gemini-3.5-flash-lite` | Model sinh câu trả lời. |
| `GEMINI_FALLBACK_MODELS` | `["gemini-3.1-flash-lite"]` | Tối đa một fallback; tổng tối đa hai model. |
| `DATABASE_URL` | rỗng | Chuỗi kết nối PostgreSQL. |
| `SIMILARITY_THRESHOLD` | `0.5` | Ngưỡng similarity tối thiểu để gọi LLM. |
| `TOP_K_DEFAULT` | `3` | Số chunk được retrieve mặc định. |
| `SEMANTIC_BREAKPOINT_PERCENTILE` | `90` | Mức percentile dùng để phát hiện điểm chuyển chủ đề. |
| `MIN_CHUNK_SIZE` | `350` | Kích thước mục tiêu tối thiểu trước khi ngắt theo ngữ nghĩa. |
| `MAX_CHUNK_SIZE` | `1200` | Kích thước tối đa của chunk. |
| `CHUNK_OVERLAP` | `150` | Số ký tự tối đa của các câu hoàn chỉnh được gối sang chunk tiếp theo. |
| `MAX_CONTEXT_TOKENS` | `1048576` | Giới hạn context window của model (tokens). Prompt sẽ được cắt chunk nếu vượt. |
| `MAX_OUTPUT_TOKENS` | `2048` | Số token tối đa cho câu trả lời. Tăng nếu câu trả lời bị cụt. |
| `LOG_LEVEL` | `DEBUG` | Mức log của service. |

Startup yêu cầu `APP_ENV`, `INTERNAL_API_KEY`, `GEMINI_INGESTION_API_KEY` và `GEMINI_QUERY_API_KEY`
đều không rỗng. Thiếu biến nào sẽ dừng trước khi kết nối DB, báo rõ tên biến;
không in giá trị key. Không hỗ trợ key chung hoặc lấy key của luồng bên kia.
Restart service sau khi đổi môi trường; cấu hình và client được cache trong process.

### CORS

Các giá trị CORS được đọc từ `.env`; danh sách dùng JSON array:

```env
CORS_ALLOW_ORIGINS=["http://localhost:3000"]
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=["POST","GET"]
CORS_ALLOW_HEADERS=["*"]
```

Default giữ hành vi trước đây. Đặt `CORS_ALLOW_ORIGINS=[]` nếu không cần cho phép
trình duyệt gọi khác origin. CORS là chính sách của trình duyệt, không chặn request
BE → RAG và không thay thế `X-Internal-Key`. Không đưa internal key vào frontend.
Thêm `DELETE` vào danh sách methods nếu cần preflight cho method đó. Restart RAG
sau khi đổi cấu hình. Xem [tài liệu FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/).

### Deadline và retry

- Chat tổng tối đa 55 giây: embed câu hỏi + retrieval có ngân sách chung 10 giây;
  mỗi model tối đa 15 giây, bao gồm đếm token, cắt context và generation.
- Mặc định gọi `gemini-3.5-flash-lite`, rồi fallback `gemini-3.1-flash-lite`.
  `GEMINI_FALLBACK_MODELS` là JSON array tối đa một tên model fallback, tổng
  tối đa hai model. Đặt `[]` để tắt fallback. Chỉ thử model tiếp theo khi lượt trước thất bại;
  dừng ngay khi có câu trả lời hoặc hết deadline tổng.
- `INGESTION_EMBED_TIMEOUT_SECONDS` mặc định 20 giây cho từng request embed.
  Retry tối đa ba lần sau lần đầu, chỉ cho 503 và 429 tạm thời. Quota ngày/quota
  bằng 0, lỗi billing, 400/401/403 và timeout không được retry.
- SDK dùng HTTP timeout riêng cho từng request và tắt retry ngầm (`attempts=1`).
  Embedding dùng SDK async, không chạy request trong `asyncio.to_thread`.
  Deadline ngoài giới hạn toàn bước; HTTP timeout giới hạn request thực.

Timeout SDK được truyền bằng milliseconds theo [Google Gen AI SDK](https://googleapis.github.io/python-genai/genai.html).

## 4. Khởi tạo vector store

Chạy migration sau trên PostgreSQL/Supabase SQL Editor:

```text
migrations/001_create_document_chunks.sql
```

Migration sẽ:

- bật extension `vector`;
- tạo bảng `document_chunks`;
- tạo vector column 768 chiều;
- tạo IVFFlat index dùng cosine distance;
- tạo unique constraint `(source_file, chunk_index)`.

## 5. Nạp tài liệu

CLI ingestion từ file/thư mục local đã được loại bỏ. Nạp tài liệu qua trang quản trị:

1. Admin upload file gốc lên Cloudinary theo upload intent của BE.
2. BE xác nhận upload và đưa tài liệu vào hàng đợi ingestion.
3. Worker BE tải file và gửi binary đến `POST /v1/documents/{document_id}/ingest`.
4. RAG kiểm tra/ghi nhận attempt trong transaction ngắn, trả connection về pool,
   rồi đọc file tạm, chia chunks và tạo embeddings bên ngoài transaction.
5. RAG lấy advisory lock lại, đọc lại state, rồi thay chunks và cập nhật index
   atomically trong transaction ngắn nếu attempt vẫn hợp lệ.

Sau khi đọc tài liệu, `sanitize_text()` loại bỏ NUL (`U+0000`) và thay surrogate
không hợp lệ bằng ký tự thay thế `U+FFFD`, trước cả chunking và embedding. Unicode
tiếng Việt, emoji, khoảng trắng và Markdown/LaTeX hợp lệ được giữ nguyên. Text
được embed cũng chính là text lưu DB; tài liệu rỗng sau làm sạch bị từ chối.
Log làm sạch chỉ ghi số ký tự trước/sau, không ghi nội dung tài liệu.

Luồng managed tiếp tục dùng UUID/attempt, advisory lock và tombstone. File tạm
được dọn sau xử lý; không cần đặt tài liệu vào thư mục local để ingest.

`rag_index_state.attempt` ghi nhận attempt mới nhất đã được chấp nhận;
`chunk_count=0` biểu thị attempt đó chưa hoàn tất. Nếu chuẩn bị thất bại hoặc
bị hủy, chunks cũ còn nguyên và có thể retry cùng attempt. Attempt cũ hơn bị
chặn kể cả khi attempt mới chưa ghi xong. Delete có thể ghi tombstone trong
lúc embed; pha ghi bắt buộc kiểm tra lại tombstone trước khi sửa chunks.
Hai replica có thể chuẩn bị cùng attempt, nhưng chỉ kết quả ghi đầu tiên được
chấp nhận; request còn lại trả chunk count đã commit. Semaphore vẫn giới hạn
chuẩn bị một tài liệu/process, không giữ connection hay SQL lock khi gọi Gemini.

Thay đổi này không cần migration hay thay đổi BE. Rollout/rollback bằng phiên bản
RAG tương ứng, giữ nguyên state và tombstone; không reset hoặc re-index DB.

Không hỗ trợ endpoint upload legacy hay CLI. Không cần mount thư mục tài liệu
local vào Docker. Việc loại bỏ luồng cũ không xóa dữ liệu DB hay file trên máy.

## 6. Chạy service

```powershell
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Các URL trong môi trường development:

- Health check: `http://localhost:8000/v1/health`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

Kiểm tra health:

```powershell
Invoke-RestMethod http://localhost:8000/v1/health
```

## 7. Gọi API chat trực tiếp

Endpoint:

```http
POST /v1/chat
X-Internal-Key: <INTERNAL_API_KEY>
Content-Type: application/json
```

Ví dụ PowerShell:

```powershell
$headers = @{ "X-Internal-Key" = "replace-with-your-secret" }
$body = @{
    question = "Cách xử lý khi pH ao nuôi biến động mạnh?"
    user_id = "00000000-0000-0000-0000-000000000001"
    conversation_id = $null
    season_id = $null
    top_k = 5
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri http://localhost:8000/v1/chat `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $body
```

Request body:

| Trường | Kiểu | Bắt buộc | Ghi chú |
|---|---|---:|---|
| `question` | string | Có | Từ 1 đến 2000 ký tự. |
| `user_id` | UUID | Có | ID người đặt câu hỏi; hiện dùng để log. |
| `conversation_id` | UUID hoặc `null` | Không | ID hội thoại; hiện dùng để log. |
| `season_id` | UUID hoặc `null` | Không | ID vụ nuôi; hiện chưa dùng để lọc retrieval. |
| `top_k` | integer hoặc `null` | Không | Từ 1 đến 20; bỏ trống để dùng `TOP_K_DEFAULT`. |
| `eligible_documents` | array hoặc `null` | Không | BE cung cấp các cặp `document_id`/`attempt` của tài liệu READY + active. Thiếu/null/rỗng sẽ không truy xuất nguồn. |

Response mẫu:

```json
{
  "answer": "Cần kiểm tra độ kiềm, mật độ tảo và đo pH vào sáng sớm, buổi chiều...",
  "query_status": "answered",
  "retrieved_chunks": [
    {
      "source_file": "huong-dan-nuoi-tom.pdf",
      "chunk_index": 4,
      "content": "Nội dung đoạn tài liệu được sử dụng...",
      "similarity": 0.82,
      "metadata": {
        "format": "pdf",
        "pages": 20,
        "chunking_strategy": "semantic",
        "chunk_size_chars": 934
      }
    }
  ],
  "top_similarity": 0.82,
  "model_version": "gemini-3.5-flash-lite",
  "processing_time_ms": 2100
}
```

### `query_status`

| Trạng thái | Ý nghĩa |
|---|---|
| `answered` | Có chunk đạt ngưỡng và Gemini đã sinh câu trả lời. |
| `low_match` | Similarity thấp hơn ngưỡng; vẫn sinh câu trả lời, backend kèm cảnh báo cần kiểm chứng. |
| `no_source` | Không có chunk; vẫn sinh câu trả lời tổng quát, backend cảnh báo chưa có nguồn tham chiếu. |
| `error` | Lỗi embedding, database, Gemini hoặc lỗi kỹ thuật khác. |

### Mã HTTP đáng chú ý

| HTTP status | Trường hợp |
|---:|---|
| `200` | Pipeline chạy xong; kết quả nghiệp vụ nằm trong `query_status`. |
| `401` | `X-Internal-Key` sai. |
| `422` | Thiếu header, UUID không hợp lệ, câu hỏi rỗng hoặc `top_k` ngoài khoảng. |
| `500` | Lỗi chưa được pipeline xử lý. |

## 8. Chạy kiểm thử

```powershell
pytest tests -v
```

Các test semantic chunking sử dụng embedding giả, không gọi Gemini thật.

## 9. Chạy bằng Docker

Chạy từ thư mục gốc `smartshrimp_ai/`. Compose chung quản lý hai service
`classification` và `rag`; có thể chỉ khởi động RAG bằng lệnh sau:

Hai image dùng Python 3.12.12. Khi chạy local, dùng Python 3.12 và virtual
environment riêng trong `rag/`; không cài chung dependency với classification.

```powershell
docker compose up --build -d --wait rag
```

Xem log:

```powershell
docker compose logs -f rag
```

Dừng service:

```powershell
docker compose stop rag
```

Healthcheck trong Compose gốc gọi `/v1/health` bằng Python. Cổng local là `8000`;
classification dùng cổng `8080`. Không cần cài `curl` trong image.

## 10. Cấu trúc thư mục

```text
smartshrimp_ai/rag/
├── app/
│   ├── api/v1/endpoints/     # Health và chat endpoints
│   ├── core/                 # Embedder, retriever, generator, pipeline
│   ├── db/                   # PostgreSQL pool và vector store
│   ├── ingestion/            # Loader, semantic chunker, API ingestion
│   ├── models/               # Request/response schemas
│   ├── config.py             # Environment settings
│   ├── dependencies.py       # X-Internal-Key guard
│   └── main.py               # FastAPI application
├── tests/                    # Unit và API tests
├── .env                      # Cấu hình local, không commit
├── Dockerfile
└── requirements.txt
```

---

# Phần 2 — Tích hợp với hệ thống SmartShrimp

## 1. Phân chia trách nhiệm

```text
Mobile/Web App
     │ JWT của người dùng
     ▼
smartshrimp_be
     │ xác thực, phân quyền, quản lý hội thoại, lưu lịch sử
     │ X-Internal-Key
     ▼
smartshrimp_rag
     │ embed → retrieve → generate
     ▼
PostgreSQL/pgvector + Gemini
```

### `smartshrimp_be` chịu trách nhiệm

- Xác thực JWT của người dùng.
- Kiểm tra người dùng có quyền truy cập `season_id` hay không.
- Tạo hoặc kiểm tra `rag_conversations`.
- Gọi RAG service bằng internal key.
- Lưu request và kết quả vào `rag_queries`.
- Trả response phù hợp cho mobile/web.
- Quản lý feedback trong `rag_feedback`.

### `smartshrimp_rag` chịu trách nhiệm

- Xác thực `X-Internal-Key` giữa hai service.
- Embed câu hỏi.
- Tìm top-K chunk trong `document_chunks`.
- Kiểm tra similarity threshold.
- Sinh câu trả lời dựa trên tài liệu; khi thiếu nguồn, trả lời tổng quát và yêu cầu kiểm chứng.
- Trả kết quả retrieval và thông tin xử lý.

RAG service không xác thực JWT và không nên được mobile/web gọi trực tiếp.

## 2. Cấu hình giữa hai service

Hai service phải dùng cùng một internal key.

Trong `smartshrimp_ai/rag/.env`:

```env
APP_ENV=development
INTERNAL_API_KEY=a-long-random-shared-secret
```

Các biến sau cần được bổ sung vào `smartshrimp_be/.env` khi triển khai connector:

```env
RAG_SERVICE_URL=http://localhost:8000
RAG_INTERNAL_API_KEY=a-long-random-shared-secret
RAG_REQUEST_TIMEOUT_MS=60000
```

Khi chạy hai service trong cùng Docker network, URL nên dùng service name, ví dụ:

```env
RAG_SERVICE_URL=http://rag:8000
```

Không đưa `RAG_INTERNAL_API_KEY` vào source code, mobile app hoặc frontend.

## 3. Contract gọi từ backend sang RAG

Backend gọi:

```http
POST {RAG_SERVICE_URL}/v1/chat
X-Internal-Key: {RAG_INTERNAL_API_KEY}
Content-Type: application/json
```

Body:

```json
{
  "question": "Dấu hiệu bệnh đục cơ trên tôm là gì?",
  "user_id": "0e563d8b-a3f2-43c9-af85-7b3b58221230",
  "conversation_id": "2c810541-4744-4767-b9dd-6ed05a4f15d5",
  "season_id": null,
  "top_k": 5
}
```

Ví dụ connector dùng `axios` trong `smartshrimp_be`:

```js
const axios = require('axios');

async function askRag(payload) {
    const baseUrl = process.env.RAG_SERVICE_URL.replace(/\/+$/, '');
    const timeout = Number(process.env.RAG_REQUEST_TIMEOUT_MS || 60000);

    const response = await axios.post(`${baseUrl}/v1/chat`, payload, {
        timeout,
        headers: {
            'Content-Type': 'application/json',
            'X-Internal-Key': process.env.RAG_INTERNAL_API_KEY,
        },
    });

    return response.data;
}
```

Đây là ví dụ contract, chưa phải file đang tồn tại trong `smartshrimp_be`.

## 4. Luồng xử lý đề xuất ở backend

1. Client gửi câu hỏi tới endpoint chat của `smartshrimp_be` kèm JWT.
2. Backend lấy `account_id` từ access token; không tin `user_id` do client gửi.
3. Nếu có `season_id`, backend kiểm tra quyền truy cập vụ nuôi.
4. Backend tạo hội thoại mới hoặc xác minh `conversation_id` thuộc người dùng.
5. Backend gọi `POST /v1/chat` của RAG service.
6. Backend lưu kết quả vào `rag_queries`.
7. Backend cập nhật `rag_conversations.last_message_at`.
8. Backend trả câu trả lời và `query_id` cho client để có thể gửi feedback.

Không để client tự truyền internal key hoặc gọi thẳng cổng `8000`.

## 5. Ánh xạ response sang database SmartShrimp

Schema SQL hiện có các bảng `rag_conversations`, `rag_queries` và `rag_feedback`.
Kết quả RAG có thể ánh xạ vào `rag_queries` như sau:

| Response RAG | Cột `rag_queries` |
|---|---|
| request `conversation_id` | `conversation_id` |
| request `user_id` | `account_id` |
| request `question` | `question` |
| `answer` | `answer` |
| `query_status` | `query_status` |
| `retrieved_chunks` | `retrieved_chunks` |
| `top_similarity` | `top_similarity` |
| `model_version` | `model_version` |
| `processing_time_ms` | `processing_time_ms` |

Lưu ý khi tích hợp:

- Nhánh `feat/85-87-rag-conversations-feedback` của backend đã bổ sung Prisma models
  ánh xạ các bảng RAG hiện có trong `smartshrimp.sql`. Chạy `npx prisma generate`
  trước khi khởi động backend; không cần thay đổi database.
- Constraint của `rag_queries` yêu cầu `error_message` khi `query_status='error'`,
  trong khi `ChatResponse` hiện không trả `error_message`. Backend cần tạm lưu một
  thông báo lỗi chuẩn hóa, hoặc contract RAG cần được mở rộng trước khi hoàn thiện
  nhánh lỗi.
- `user_id`, `conversation_id` và `season_id` hiện chỉ được RAG service ghi log;
  retrieval chưa lọc theo người dùng, hội thoại hoặc vụ nuôi.

## 6. Xử lý lỗi khi tích hợp

Backend nên phân biệt hai nhóm lỗi:

### RAG trả HTTP 200

Đọc `query_status`:

- `answered`: lưu và trả câu trả lời.
- `low_match`: lưu và trả câu trả lời cùng nguồn đã retrieve; cảnh báo độ liên quan thấp.
- `no_source`: lưu và trả câu trả lời tổng quát; cảnh báo chưa có nguồn tham chiếu.
- `error`: lưu lỗi chuẩn hóa; trả thông báo tạm thời không thể xử lý.

### RAG không trả HTTP 200

- `401`: cấu hình internal key giữa hai service không khớp.
- `422`: payload backend gửi sang RAG không hợp lệ.
- timeout, mất kết nối hoặc `5xx`: ghi log kỹ thuật ở backend và trả lỗi dịch vụ
  tạm thời cho client.

Không trả Gemini API key, internal key, stack trace hoặc chi tiết kết nối database
cho client.

## 7. Checklist tích hợp SmartShrimp

Backend API và Flutter đã được triển khai trên nhánh `feat/85-87-rag-conversations-feedback`
trong từng repository. Xem `smartshrimp_be/docs/rag-api.md` để biết contract, quyền,
cấu hình và kết quả kiểm thử. Checklist dưới đây là hướng dẫn triển khai môi trường thực.

- [ ] Thêm cấu hình RAG vào `smartshrimp_be/src/config/index.js`.
- [ ] Thêm service gọi `POST /v1/chat` bằng `axios`.
- [ ] Thêm validator cho câu hỏi, `conversation_id`, `season_id` và `top_k`.
- [ ] Thêm route/controller chat có middleware xác thực người dùng.
- [ ] Thêm Prisma models hoặc data-access cho các bảng RAG.
- [ ] Kiểm tra quyền sở hữu conversation và quyền truy cập season.
- [ ] Lưu mọi trạng thái `answered`, `low_match`, `no_source`, `error`.
- [ ] Thêm endpoint gửi feedback vào `rag_feedback`.
- [ ] Không public trực tiếp RAG service ra mobile/web.
- [ ] Thêm integration test mock RAG service và test timeout/lỗi kết nối.

## 8. Kiểm tra sau khi tích hợp

1. Gọi `GET /v1/health` từ môi trường chạy backend.
2. Gọi RAG trực tiếp bằng internal key để xác nhận contract.
3. Gọi endpoint chat của `smartshrimp_be` bằng JWT hợp lệ.
4. Kiểm tra bản ghi `rag_conversations` và `rag_queries`.
5. Kiểm tra các nhánh `low_match`, `no_source`, `error`.
6. Xác nhận mobile/web không biết và không gửi `X-Internal-Key`.

