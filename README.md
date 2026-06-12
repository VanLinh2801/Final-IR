# Student RAG Server

FastAPI server cho bài thi RAG offline với 2 endpoint bắt buộc:
- `POST /upload`
- `POST /ask`

Server dùng:
- embedding local `keepitreal/vietnamese-sbert`
- retrieval trong RAM bằng `numpy`
- teacher proxy theo chuẩn OpenAI-compatible API
- local persistence cho retrieval index tại `storage/index`

## 1. Chuẩn bị môi trường

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Cấu hình

Tạo file `.env` từ `.env.example` rồi điền:
- `STUDENT_ID`
- `EMBEDDING_MODEL_PATH`
- `INDEX_STORAGE_DIR`
- `SERVER_PORT`
- nếu cần, `SERVER_PUBLIC_IP`

`SERVER_PUBLIC_IP` dùng để override IP tự dò khi máy có nhiều card mạng hoặc VPN.

## 3. Chạy server

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

`/health` sẽ trả thêm:
- `rag_ready`
- `doc_id`
- `chunk_count`
- `index_persisted`

## 4. Quy trình thi mới

Lần đầu khi vào thi:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --document-received false
```

Khi teacher server gửi tài liệu sang `/upload`, server sẽ:
- chunking văn bản
- tạo embeddings
- lưu index local vào `storage/index`

Sau khi upload xong, kiểm tra:

```powershell
curl http://127.0.0.1:8000/health
```

Chỉ tiếp tục nộp lại khi:
- `rag_ready=true`
- `index_persisted=true`

Những lần evaluate sau:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --document-received true
```

Nếu tắt server hoặc restart máy, server sẽ tự load lại index từ `storage/index` khi khởi động lại.

## 5. Gọi helper scripts

Đăng ký server lên teacher server:

```powershell
.\.venv\Scripts\python.exe scripts/register.py
```

Kiểm tra trạng thái:

```powershell
.\.venv\Scripts\python.exe scripts/result.py
```

Reset trạng thái trên teacher server:

```powershell
.\.venv\Scripts\python.exe scripts/reset.py
```

Lưu ý: `register.py` và `reset.py` mặc định không xóa index local.

## 6. Lưu ý về persistence

Persistence chỉ hợp lệ khi các cấu hình sau không đổi:
- `EMBEDDING_MODEL_PATH`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

Nếu thay đổi một trong các cấu hình trên, server sẽ bỏ qua index cũ và cần upload/build lại.

## 7. Checklist trước khi thi

- xác nhận model local load được khi ngắt mạng
- chạy server bằng IP LAN mà teacher server truy cập được
- không đăng ký `localhost` hoặc `127.0.0.1`
- `POST /ask` luôn trả đúng một ký tự `A/B/C/D`
- nếu tự dò IP sai thì set `SERVER_PUBLIC_IP` trong `.env`
- sau lần upload đầu tiên, kiểm tra `storage/index` đã có `metadata.json` và `embeddings.npy`
