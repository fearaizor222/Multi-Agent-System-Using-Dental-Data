# Kho dữ liệu và trợ lý nha khoa local

Thu thập và xử lý ba nguồn: PDF Bộ Y tế, MSD tiếng Việt và bài toàn văn PMC.
Code hiện có phần tạo kho dữ liệu, chỉ mục truy xuất và luồng thử nghiệm **A1 → A2 → A3 → A4**. A1–A3 dùng model local; A4 tổng hợp theo quy tắc từ các nhận định được hỗ trợ. Chạy trợ lý hoặc chat nhiều lượt theo [hướng dẫn A4 và cách chạy](docs/a4_synthesis.md). A5 và công cụ dịch vụ đang tắt.

## Dữ liệu hiện có

| Nguồn | Phạm vi | Số tài liệu | Số chunk |
|---|---|---:|---:|
| Bộ Y tế | Trang PDF 8–201, gồm 57 chương của hướng dẫn năm 2015 | 1 | 3.005 |
| MSD tiếng Việt | Mục Rối loạn nha khoa, bản chuyên gia | 46 | 1.285 |
| PMC | Toàn văn Open Access, chia theo 12 nhóm chủ đề | 200 | 19.121 |
| Tổng | | 247 | 23.411 |

Đây là kết quả đợt thu thập ngày 15/10/2026. Các bài PMC tiếng Anh được giữ nguyên.
Kho đã có embedding/chỉ mục và được nối vào luồng tạo nháp/đối chiếu nguồn. Xem [truy xuất](docs/retrieval.md) và [A2–A3](docs/agents_and_reranking.md). Phần dưới mô tả quy trình thu thập và chuẩn hóa kho dữ liệu.

## Cấu trúc thư mục

```text
config/
    pmc_topics.json                 Chủ đề và câu truy vấn tìm bài PMC
    schemas/                        JSON Schema của tài liệu và chunk
scripts/
    collect_public_bulk.py          Thu thập hàng loạt MSD hoặc PMC
    process_byt_pdf.py              Đọc PDF Bộ Y tế
    reprocess_public_bulk.py        Xử lý lại từ bản gốc, không tải lại
    collect_table_images.py         Lưu ảnh của các bảng PMC không có ô văn bản
    check_public_bulk.py            Kiểm tra dữ liệu và chia chunk
    export_public_catalog.py        Xuất danh sách và bản đọc từng tài liệu
src/data/
    bulk_public.py                  Tìm bài, tải, cache và ghi tiến độ
    public_schema.py                Cấu trúc tài liệu, mã băm và phiên bản
    public_chunks.py                Chia đoạn, giữ liên kết về nguồn
    public_catalog.py               Hiển thị nội dung tài liệu đã xử lý
    sources/
        byt.py                      Bộ đọc PDF Bộ Y tế
        msd.py                      Bộ đọc HTML MSD
        msd_http.py                 URL, robots và tải HTML MSD
        pmc.py                      Bộ đọc JATS/XML PMC
        markup.py                   Xử lý văn bản và bảng HTML/XML
data/raw/public_v1/                 PDF, HTML, XML, ảnh bảng và cache
data/processed/public_v1/corpus/    Tài liệu chuẩn hóa, phiên bản và chunk
outputs/public_v1_bulk/             Danh sách, bản đọc và kết quả kiểm tra
tests/                             Kiểm tra bộ đọc, chunk và thu thập
```

## Môi trường

Python 3.11. Môi trường ảo trên máy hiện tại nằm ở thư mục DATN cha.
Mở terminal tại thư mục chứa README này:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Chunk sử dụng tokenizer `intfloat/multilingual-e5-small`. Máy hiện tại đã có cache;
trên máy khác tải tokenizer một lần trước khi kiểm tra kho:

```powershell
..\.venv\Scripts\python.exe -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')"
```

Không cần API key hay model sinh câu trả lời để chạy riêng phần thu thập/xử lý dữ liệu này. Chạy trợ lý cần các model local và hai môi trường đã nêu trong hướng dẫn tích hợp.

## Chạy từ đầu

### 1. Thu thập và chuẩn hóa từng nguồn

```powershell
..\.venv\Scripts\python.exe scripts\process_byt_pdf.py "..\Bản-cuối-20150728-HD-CĐĐT-RHM.pdf"
..\.venv\Scripts\python.exe -u scripts\collect_public_bulk.py msd
..\.venv\Scripts\python.exe -u scripts\collect_public_bulk.py pmc --target 200
```

PDF đầu vào hiện nằm ở DATN cha. Khi chuyển máy, có thể truyền đường dẫn PDF khác.
MSD và PMC giữ cache để tiếp tục khi chạy lại. PMC tìm metadata qua Europe PMC,
còn toàn văn tải từ NIH/NLM PMC Cloud. Khoảng ngày tìm bài hiện đặt đến 15/10/2026.

### 2. Lưu ảnh bảng, kiểm tra và chia đoạn

```powershell
..\.venv\Scripts\python.exe scripts\collect_table_images.py
..\.venv\Scripts\python.exe -u scripts\check_public_bulk.py
```

Chỉ khi kiểm tra đạt, script mới ghi file `chunks.jsonl` chính thức.
Mỗi chunk tối đa 380 token, tính cả tên tài liệu, đề mục và `passage:`.

### 3. Xuất bản xem dữ liệu

```powershell
..\.venv\Scripts\python.exe scripts\export_public_catalog.py
```

Mở `outputs/public_v1_bulk/index.html`, bấm tên tài liệu để đọc nội dung đã xử lý.
`documents.csv` là danh sách tài liệu; `review_items.csv` liệt kê các bảng cần rà lại.

## Xử lý lại khi sửa bộ đọc

```powershell
..\.venv\Scripts\python.exe -u scripts\reprocess_public_bulk.py
```

Lệnh này đọc lại các bản gốc đã lưu, không gọi mạng. Có thể thêm `--source byt`,
`--source msd` hoặc `--source pmc`. Sau đó chạy lại bước lưu ảnh bảng, kiểm tra và xuất
bản xem để các kết quả khớp với phiên bản tài liệu mới.

## Chạy tests

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Tests không gọi mạng và không tải lại kho dữ liệu. Fixture XML trong tests là dữ
liệu tự tạo để kiểm tra bộ đọc, không đưa vào kho tài liệu thật.

## Những phần cần rà lại

- 29 khối bảng PDF có tiêu đề cột được suy ra từ bố cục.
- 14 bảng PMC chỉ có ảnh: đã lưu ảnh gốc, chưa OCR phần chữ/số trong ảnh.
- Chú thích hình được giữ; nội dung bên trong hình chưa được phân tích.
- Kiểm tra tự động xác nhận cấu trúc và toàn vẹn dữ liệu; chưa đánh giá độ đúng
  chuyên môn của từng tài liệu.

Chi tiết: [quy trình thu thập](docs/public_bulk.md) và
[cấu trúc tài liệu, chunk](docs/public_data_schema.md).

Code, cấu hình và hướng dẫn dùng để đưa lên Git. Dữ liệu tải về, kết quả xử lý,
bản xem và môi trường ảo đều được Git bỏ qua.

## Web thử nghiệm bốn agent

```powershell
..\.venv\Scripts\python.exe -X utf8 -m scripts.run_web
```

Mở http://127.0.0.1:8000. A1 và A2 dùng Qwen3 1.7B với hai adapter đã fine tune;
A3 và A4 dùng Qwen3 1.7B gốc. Chưa tích hợp xử lý ảnh.
Chi tiết cấu hình, cách chạy và vị trí kết quả: [web thử nghiệm](docs/web_test.md).
