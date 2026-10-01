# 🦷 Multi-Agent System Using Dental Data with LangGraph & Hybrid RAG

Hệ thống **Multi-Agent** chuyên khoa Răng Hàm Mặt xây dựng trên nền tảng **LangChain**, **LangGraph** và định hướng tích hợp **Hybrid RAG** (Dense Vector + Sparse BM25 với thuật toán Reciprocal Rank Fusion - RRF).

Dự án được cấu trúc theo tiêu chuẩn mã nguồn chuyên nghiệp, có tính module hóa cao, phù hợp cho **Đồ án tốt nghiệp / Nghiên cứu chuyên sâu về Hệ thống Đa Tác tử trong Y tế**.

---

## 🏛️ 1. Kiến trúc Hệ Thống (Architecture)

### Mô hình Multi-Agent Supervisor Pattern
Hệ thống sử dụng mô hình điều phối phân cấp (**Supervisor Pattern**). Trưởng nhóm điều phối (`Supervisor Agent`) phân tích truy vấn của bệnh nhân và định tuyến luồng xử lý tới các Agent chuyên trách:

```mermaid
flowchart TD
    START([Bắt đầu / Bệnh nhân hỏi]) --> Supervisor[👔 Supervisor Agent]
    
    Supervisor -->|Cần tra cứu y văn| RAG[🔍 RAG Retriever Agent\nHybrid Search: Vector + BM25]
    Supervisor -->|Đã có dữ liệu y khoa| Consultant[🩺 Dental Consultant Agent\nBác sĩ tư vấn lâm sàng]
    
    RAG -->|Trả về tài liệu y khoa| Supervisor
    Consultant -->|Trả lời hoàn tất| Supervisor
    
    Supervisor -->|Kết thúc luồng| END([Hoàn thành & Trả lời bệnh nhân])

    subgraph Hybrid RAG Engine
        RAG -.-> Dense[Dense Search\nEmbeddings + VectorDB]
        RAG -.-> Sparse[Sparse Search\nBM25 Okapi]
        Dense -.-> RRF[Reciprocal Rank Fusion\n(RRF) Re-ranking]
        Sparse -.-> RRF
    end
```

### Các Agent trong hệ thống
1. **`Supervisor Agent`**: Giám sát ngữ cảnh hội thoại, quyết định thứ tự thực thi của các Worker Agent hoặc kết thúc quy trình (`FINISH`).
2. **`RAG Retriever Agent`**: Chịu trách nhiệm truy vấn kho tri thức bệnh lý nha khoa bằng cơ chế **Hybrid RAG** (kết hợp tìm kiếm ngữ nghĩa Dense và từ khóa chính xác BM25).
3. **`Dental Consultant Agent`**: Đóng vai trò bác sĩ chuyên khoa Răng Hàm Mặt, tổng hợp dữ liệu y văn từ RAG để đưa ra lời giải thích bệnh học, cảnh báo cấp cứu (Triage) và hướng dẫn bệnh nhân.

---

## 🔬 2. Cơ chế Hybrid RAG (Định hướng tương lai)

Trong lĩnh vực nha khoa, các truy vấn thường chứa:
- **Thuật ngữ chuyên môn / Số hiệu răng cụ thể**: Ví dụ `Răng số 6`, `Răng 48`, `Viêm tủy cấp`, `Composite`, `Chữa tủy nội nha`. Đây là điểm mạnh của **Sparse Search (BM25)**.
- **Mô tả triệu chứng tự nhiên theo ngôn ngữ người bệnh**: Ví dụ `uống nước đá bị buốt nhói lên tận óc`, `đánh răng hay bị chảy máu`. Đây là điểm mạnh của **Dense Vector Search (Embedding)**.

Do đó, kiến trúc **Hybrid RAG** giải quyết bài toán dung hợp bằng công thức **Reciprocal Rank Fusion (RRF)**:

$$\text{RRF\_Score}(d) = w_{\text{dense}} \cdot \frac{1}{k + \text{rank}_{\text{dense}}(d)} + w_{\text{sparse}} \cdot \frac{1}{k + \text{rank}_{\text{sparse}}(d)}$$

*(với hằng số làm mượt tiêu chuẩn $k = 60$, tỷ trọng mặc định $w_{\text{dense}} = 0.6, w_{\text{sparse}} = 0.4$)*.

---

## 📂 3. Cấu trúc Dự Án (Project Structure)

```text
DoAnTotNghiep/
├── config/
│   └── settings.yaml             # Cấu hình chi tiết các tham số Agent & RAG
├── data/
│   ├── raw/
│   │   └── dental_knowledge_sample.json # Dữ liệu tri thức nha khoa mẫu
│   ├── processed/                # Dữ liệu sau khi làm sạch / chunking
│   └── vectorstore/              # Thư mục lưu trữ ChromaDB index
├── src/
│   ├── core/
│   │   ├── config.py             # Pydantic Settings đọc cấu hình từ .env
│   │   ├── llm.py                # LLM Factory (Google Gemini, Ollama, MockLLM)
│   │   └── logger.py             # Logger hiển thị màu sắc định dạng Rich
│   ├── agents/
│   │   ├── base.py               # Node wrapper & bộ khung Agent
│   │   ├── supervisor.py         # Supervisor Agent điều phối
│   │   ├── rag_retriever_agent.py # Agent phụ trách tìm kiếm Hybrid RAG
│   │   └── dental_consultant.py  # Agent bác sĩ tư vấn nha khoa
│   ├── graph/
│   │   ├── state.py              # Schema AgentState của LangGraph
│   │   └── workflow.py           # Định nghĩa và compile StateGraph
│   ├── rag/
│   │   ├── base.py               # Interface BaseRetriever & DentalDocument
│   │   ├── bm25_retriever.py     # Sparse BM25 Keyword Search
│   │   ├── vector_store.py       # Dense Vector Search (Embeddings)
│   │   ├── hybrid_retriever.py   # Dung hợp RRF Dense + Sparse
│   │   └── sample_data.py        # Helper nạp dữ liệu nha khoa mẫu
│   └── tools/
│       └── dental_tools.py       # Công cụ đánh giá cấp cứu, tra cứu giải phẫu răng
├── tests/
│   └── test_graph.py             # Unit tests kiểm tra Graph và Hybrid RAG
├── main.py                       # CLI tương tác & Demo luồng Multi-Agent
├── .env.example                  # Template khai báo biến môi trường
├── requirements.txt              # Danh sách thư viện phụ thuộc
└── README.md
```

---

## 🚀 4. Hướng Dẫn Cài Đặt & Chạy Thử

### Bước 1: Tạo môi trường ảo Python
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
```

### Bước 3: Cấu hình API Key Google Gemini
File `.env` đã được chuẩn bị sẵn. Bạn chỉ cần đảm bảo biến `GOOGLE_API_KEY` đã được điền:
```env
LLM_PROVIDER=google
GOOGLE_MODEL_NAME=gemini-2.5-flash
GOOGLE_API_KEY=your_google_api_key_here
EMBEDDING_PROVIDER=google
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
```

> 💡 **Ghi chú**: Dự án sử dụng mô hình thế hệ mới **`gemini-2.5-flash`** cho các tác tử hội thoại và **`models/gemini-embedding-001`** cho quá trình Dense Vector Embeddings trong Hybrid RAG.

### Bước 4: Chạy chương trình
```bash
python main.py
```
Hoặc kiểm tra nhanh với câu hỏi tùy ý:
```bash
python main.py "Tôi bị đau buốt răng hàm dưới khi nhai, nướu hơi sưng thì nên xử lý thế nào?"
```

### Bước 5: Chạy kiểm thử tự động (Unit Tests)
```bash
pytest -v tests/
```

---

## 🔮 5. Định Hướng Phát Triển Tiếp Theo

1. **Bổ sung Agent Chuyên Biệt Mới**:
   - `Dental Triage Agent`: Phân loại khẩn cấp (áp-xe, chấn thương hàm mặt).
   - `Appointment Agent`: Hỗ trợ đặt lịch hẹn và kết nối với hệ thống quản lý phòng khám (HIS/EMR).
   - `Dental Imaging Agent`: Tích hợp mô hình thị giác máy tính (Vision) phân tích phim X-quang Panorama / ConeBeam CT.
2. **Nâng cấp Hybrid RAG**:
   - Tích hợp **Cross-Encoder Re-ranker** (như `bge-reranker-large` hoặc Cohere Rerank) sau bước RRF.
   - Kết nối cơ sở dữ liệu Vector Database lưu trữ phân tán (ChromaDB / Qdrant / Milvus).
   - Tích hợp Graph RAG (Knowledge Graph) biểu diễn mối quan hệ giữa Răng - Bệnh lý - Vi khuẩn - Thuốc.
