# 🦷 Multi-Agent System Using Dental Data with LangGraph, Hybrid RAG & Multimodal Vision

Hệ thống **Multi-Agent Y tế Chuyên khoa Răng Hàm Mặt** xây dựng trên nền tảng **LangChain**, **LangGraph**, tích hợp động cơ **Hybrid RAG** (Qdrant Vector Database + Sparse BM25 với thuật toán Reciprocal Rank Fusion - RRF) và mô hình thị giác nha khoa **Multimodal VLM (Dental-Qwen2-VL-2B)** được huấn luyện tinh chỉnh (QLoRA).

Dự án được thiết kế chuyên biệt phục vụ nghiên cứu lâm sàng, hệ thống điều phối đa tác tử thông minh và phân loại cấp cứu nha khoa theo thời gian thực.

---

## 🏛️ 1. Kiến trúc Hệ Thống (Architecture)

Hệ thống áp dụng mô hình điều phối phân cấp (**Hierarchical Multi-Agent Supervisor Pattern**) kết hợp đồ thị trạng thái **LangGraph StateGraph**, luân chuyển thông tin tuần tự và song song qua 5 tác tử chuyên biệt:

```mermaid
flowchart TD
    START([Bắt đầu: Bệnh nhân gửi câu hỏi & Ảnh]) --> A1[👔 Agent 1: Supervisor & Emergency Triage\nPhân loại ý định, Cờ đỏ cấp cứu]
    
    A1 -->|Có ảnh lâm sàng / X-quang| A4[👁️ Agent 4: Dental Vision VLM\nPhân tích giải phẫu FDI, tổn thương hình ảnh]
    A1 -->|Chỉ có văn bản| A2[📚 Agent 2: Closed-data RAG\nKho nội bộ: Cắn khớp học RHM & TMJ]
    
    A4 -->|VisionFindings JSON| A2
    A2 -->|Bằng chứng nội bộ I1, I2| A3[🌐 Agent 3: Open-data RAG\nKho công khai: Phác đồ Bộ Y Tế & MSD]
    
    A3 -->|Bằng chứng công khai P1, P2| A5[✍️ Agent 5: Drafting & Synthesis\nTổng hợp tư vấn lâm sàng y khoa nhân ái]
    
    A5 --> END([Hoàn tất: Trả lời bệnh nhân có trích dẫn minh bạch])

    subgraph Hybrid RAG Engine (Qdrant + BM25)
        QDRANT[(Qdrant Vector DB\nintfloat/multilingual-e5-small)]
        BM25[(BM25 Okapi\nTừ khóa chuyên ngành RHM)]
        RRF{RRF Dung hợp\nw_dense: 0.6, w_sparse: 0.4}
        QDRANT --> RRF
        BM25 --> RRF
        RRF -.-> A2
        RRF -.-> A3
    end
```

### Chi tiết 5 Tác tử Chuyên biệt (5 Specialized Dental Agents)

1. **`Agent 1: Supervisor & Emergency Triage Agent`** (`src/agents/supervisor.py`):
   - Phân tích triệu chứng, phát hiện lập tức các **cờ đỏ cấp cứu (Red Flags)**: sưng mặt lan hốc mắt/sàn miệng, khó thở/khó nuốt, sốt cao, xuất huyết ồ ạt, chấn thương rơi răng (Avulsion).
   - Tích hợp bộ quy tắc lâm sàng chuẩn hóa cùng mô hình SLM phân loại định tuyến.
2. **`Agent 2: Closed-data RAG Agent`** (`src/agents/internal_rag_agent.py`):
   - Đóng vai trò bác sĩ chuyên gia về **Cắn khớp học**, khớp thái dương hàm (TMJ) và sức khỏe răng miệng người cao tuổi.
   - Khai thác kho dữ liệu nội bộ chuẩn từ Giáo trình Chuyên khoa RHM.
3. **`Agent 3: Open-data RAG Agent`** (`src/agents/public_rag_agent.py`):
   - Đối chiếu quy trình kỹ thuật và hướng dẫn điều trị chuẩn quốc gia từ **Bộ Y Tế** và cẩm nang y khoa quốc tế **MSD Manuals**.
4. **`Agent 4: Dental Vision Agent`** (`src/agents/vision_agent.py`):
   - Phân tích ảnh lâm sàng, xác định vị trí răng theo **danh pháp quốc tế FDI** (răng 11..48), tình trạng men ngà, nướu và bệnh lý khoang miệng.
   - Trả về cấu trúc dữ liệu JSON chuẩn hóa `VisionFindings`.
5. **`Agent 5: Drafting & Synthesis Agent`** (`src/agents/drafting_agent.py`):
   - Tổng hợp toàn bộ bằng chứng y văn (`[I1]`, `[P1]`), kết quả phân tích thị giác và nhận định sơ bộ thành câu trả lời hoàn chỉnh, chu đáo, dễ hiểu và trích dẫn minh bạch.

---

## 🔬 2. Động Cơ Hybrid RAG & Vector Store

Hệ thống kết hợp sức mạnh tìm kiếm ngữ nghĩa Dense Vector và từ khóa chính xác Sparse BM25:

- **Dense Search**: Sử dụng **Qdrant Vector Database** (hỗ trợ lưu trữ cục bộ file `.sqlite` tại `data/qdrant_db/` hoặc Docker service `localhost:6333`) kết hợp mô hình embedding đa ngôn ngữ `intfloat/multilingual-e5-small`.
- **Sparse Search**: Thuật toán **BM25 Okapi** xử lý chỉ mục văn bản tiếng Việt chuyên khoa nha khoa.
- **Dung hợp RRF (Reciprocal Rank Fusion)**:
  $$\text{RRF\_Score}(d) = w_{\text{dense}} \cdot \frac{1}{k + \text{rank}_{\text{dense}}(d)} + w_{\text{sparse}} \cdot \frac{1}{k + \text{rank}_{\text{sparse}}(d)}$$
  *(với $k = 60$, tỷ trọng tối ưu: $w_{\text{dense}} = 0.6, w_{\text{sparse}} = 0.4$)*.

---

## 📂 3. Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
DoAnTotNghiep/
├── config/
│   └── settings.yaml             # Cấu hình chi tiết tham số Agent, RAG & Qdrant
├── data/
│   ├── raw/                      # Dữ liệu sách chuyên khoa & tài liệu gốc
│   ├── processed/                # Dữ liệu chunking sau tiền xử lý (internal_v1, public_v1)
│   ├── training/                 # Tập dữ liệu huấn luyện SFT chuẩn JSONL
│   │   ├── agent1_triage_dataset.jsonl   # 1,200 mẫu triage cấp cứu cho Agent 1
│   │   ├── agent5_drafting_sft.jsonl     # 550 mẫu tư vấn lâm sàng cho Agent 5
│   │   └── dental_vision_sft.jsonl       # 558 mẫu thị giác đa phương thức cho VLM
│   └── qdrant_db/                # Qdrant Vector Database cục bộ (.gitignore)
├── src/
│   ├── agents/
│   │   ├── contracts.py          # Pydantic Schemas (VisionFindings, Evidence, TriageDecision)
│   │   ├── supervisor.py         # Agent 1: Supervisor & Emergency Triage
│   │   ├── internal_rag_agent.py # Agent 2: Closed-data RAG (Cắn khớp & TMJ)
│   │   ├── public_rag_agent.py   # Agent 3: Open-data RAG (Phác đồ Bộ Y Tế)
│   │   ├── vision_agent.py       # Agent 4: Dental Vision phân tích hình thái
│   │   ├── drafting_agent.py     # Agent 5: Soạn thảo nhận định & Tư vấn lâm sàng
│   │   └── verification.py       # Tác tử kiểm chứng bằng chứng y khoa độc lập
│   ├── core/
│   │   ├── config.py             # Pydantic Settings đọc cấu hình từ .env
│   │   ├── llm.py                # LLM Factory (Gemini, Local SLMs, HuggingFace)
│   │   └── logger.py             # Hệ thống ghi log màu sắc định dạng Rich
│   ├── graph/
│   │   └── full_dental_graph.py  # LangGraph StateGraph tích hợp toàn diện 5 Agent
│   └── rag/
│       ├── base.py               # Interface BaseRetriever & DentalDocument
│       ├── bm25_retriever.py     # BM25 Keyword Search tiếng Việt
│       ├── qdrant_store.py       # Qdrant Dense Vector Store & Dense Retriever
│       └── hybrid_retriever.py   # Bộ dung hợp RRF Hybrid Retriever
├── scripts/
│   ├── prepare_vision_sft_dataset.py       # Xây dựng dataset SFT thị giác từ 3 nguồn (JSONL)
│   ├── modal_train_dental_qwen2_vl.py      # Pipeline fine-tuning QLoRA trên Modal Cloud GPU
│   ├── generate_agent1_triage_dataset.py   # Sinh dữ liệu 1,200 ca triage cấp cứu
│   ├── finetune_agent1_supervisor.py       # Pipeline fine-tuning Agent 1
│   ├── generate_agent5_drafting_dataset.py # Sinh dữ liệu 550 ca tư vấn lâm sàng cho Agent 5
│   ├── modal_train_agent5_drafting.py      # Pipeline fine-tuning QLoRA cho Agent 5 trên Modal
│   └── test_modal_dental.py                # Kiểm thử suy luận hệ thống trên Modal
├── tests/
│   ├── test_emergency_triage.py  # Bộ kiểm thử phân loại cấp cứu & cờ đỏ lâm sàng
│   └── test_graph.py             # Kiểm thử biên dịch pipeline LangGraph & Hybrid RAG
├── main_5agents.py               # CLI tương tác người bệnh trực tiếp (hỗ trợ kèm ảnh)
├── app_web.py                    # Giao diện Web / Chatbot lâm sàng (FastAPI + Tailwind SPA)
├── modal_dental_server.py        # Serverless backend triển khai SLM/VLM trên Modal Cloud
├── .env.example                  # Template biến môi trường
├── requirements.txt              # Thư viện lõi (LangChain, LangGraph, FastAPI, ...)
└── README.md
```

---

## 🚀 4. Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### Bước 1: Khởi tạo môi trường ảo Python
```bash
python -m venv venv

# Trên Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Trên Linux/macOS:
source venv/bin/activate
```

### Bước 2: Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
pip install -r requirements-embedding.txt
```

### Bước 3: Cấu hình biến môi trường
Sao chép `.env.example` thành `.env` và điền khóa API:
```bash
copy .env.example .env
```
Nội dung file `.env`:
```env
LLM_PROVIDER=google
GOOGLE_MODEL_NAME=gemini-2.5-flash
GOOGLE_API_KEY=your_google_api_key_here

EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-small

QDRANT_STORAGE_PATH=./data/qdrant_db
QDRANT_URL=http://localhost:6333
```

---

## 💻 5. Cách Thức Vận Hành

### 1. Chạy CLI 5-Agent Pipeline
Tương tác trực tiếp bằng giao diện dòng lệnh:
```bash
python main_5agents.py
```
Hoặc kiểm tra nhanh một câu hỏi kèm hình ảnh bệnh nhân:
```bash
python main_5agents.py "Răng hàm dưới của tôi bị buốt nhói khi ăn đồ ngọt, nướu bị sưng đỏ" --image "data/images/dental_public_health/fig_19_01_gingivitis.jpg"
```

### 2. Khởi chạy Giao Diện Web (FastAPI + Tailwind SPA)
Khởi chạy web server ứng dụng tư vấn nha khoa:
```bash
python app_web.py
```
Mở trình duyệt tại: **`http://localhost:8000`** để chat trực tiếp và tải ảnh răng miệng lên phân tích.

### 3. Chạy Serverless Backend trên Modal Cloud GPU
Triển khai hệ thống 5 Small Language Models (SLMs) chuyên biệt trên GPU T4/A10G của Modal:
```bash
modal serve modal_dental_server.py
```

### 4. Chạy Bộ Kiểm Thử Tự Động (Unit Tests)
```bash
pytest -v tests/test_graph.py tests/test_emergency_triage.py
```

---

## 🧠 6. Huấn Luyện Mô Hình Tinh Chỉnh (Fine-tuning & SFT)

### Huấn luyện Agent 1 (Supervisor & Emergency Triage)
Sinh tập dữ liệu 1,200 ca bệnh đa dạng và huấn luyện SLM phân loại:
```bash
python scripts/generate_agent1_triage_dataset.py
python scripts/finetune_agent1_supervisor.py
```

### Huấn luyện Tác tử Thị giác Nha khoa (Dental-Qwen2-VL-2B)
Tổng hợp 558 mẫu SFT đa nguồn và huấn luyện QLoRA trên Modal Cloud GPU (A10G/T4):
```bash
# Bước 1: Tổng hợp và chuẩn hóa dataset SFT (JSONL)
python scripts/prepare_vision_sft_dataset.py

# Bước 2: Huấn luyện QLoRA trên Modal Cloud
modal run scripts/modal_train_dental_qwen2_vl.py
```
Mô hình sau khi huấn luyện sẽ tự động lưu LoRA Adapter vào volume Modal và có thể tải về để suy luận offline tại `models/dental_qwen2_vl_adapter/`.

### Huấn luyện Agent 5 (Drafting & Consultation Synthesis)
Tổng hợp 550 mẫu SFT tư vấn lâm sàng và huấn luyện QLoRA trên Modal Cloud GPU (A10G):
```bash
# Bước 1: Sinh tập dữ liệu SFT 550 ca bệnh có trích dẫn [I1], [P1]
python scripts/generate_agent5_drafting_dataset.py

# Bước 2: Huấn luyện QLoRA Qwen2.5-1.5B trên Modal Cloud
modal run scripts/modal_train_agent5_drafting.py
```
Trọng số LoRA Adapter lưu tại volume `dental-agent5-volume` và tải về cục bộ tại `models/dental_agent5_drafting_adapter/`.

---

## ⚖️ Giấy Phép & Khuyến Cáo Y Khoa

- **Khuyến cáo y tế**: Hệ thống được phát triển nhằm mục đích nghiên cứu học thuật và hỗ trợ tham vấn sơ bộ. Kết quả từ hệ thống **không thay thế** cho chẩn đoán chuyên môn, chỉ định lâm sàng hay đơn thuốc của bác sĩ Răng Hàm Mặt có chứng chỉ hành nghề.
- **Giấy phép**: Đồ án tốt nghiệp / Mã nguồn mở phục vụ nghiên cứu giáo dục.
