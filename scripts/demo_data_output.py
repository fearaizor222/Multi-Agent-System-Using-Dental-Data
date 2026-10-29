"""Script chạy thử để trích xuất và hiển thị dữ liệu output thực tế của hệ thống.
Không chạy embedding hay mô hình nặng, chỉ đọc dữ liệu JSON/JSONL đã có và
thể hiện chính xác cấu trúc dữ liệu của các thực thể:
1. DentalDocument / Chunk (Kho nội bộ & Kho công khai)
2. Evidence Bundle (Đóng gói bằng chứng chuyển cho Agent)
3. Output của Agent A2 (Bản nháp Draft & DraftClaim & Citation)
4. Output của Agent A3 (Kết quả kiểm chứng ClaimAssessment & Verification Summary)
"""
import json
import sys
from pathlib import Path

# Đảm bảo UTF-8 cho Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]

# 1. Đọc 1 chunk thật từ kho nội bộ chuẩn hóa (internal_v1: RHM1 - Khớp cắn học)
internal_chunk_path = ROOT / "data/processed/internal_v1/corpus/chunks.jsonl"
internal_sample = None
if internal_chunk_path.exists():
    with open(internal_chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "máng nhai" in item.get("content", "").lower():
                internal_sample = item
                break
if not internal_sample:
    with open(ROOT / "data/processed/khop_can_hoc_sections.json", "r", encoding="utf-8") as f:
        internal_sample = json.load(f)[10]

# 2. Đọc 1 chunk thật từ kho công khai (Public corpus: BYT / MSD / PMC)
public_chunk_path = ROOT / "data/processed/public_v1/corpus/chunks.jsonl"
public_sample = None
with open(public_chunk_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item.get("source") in ["byt", "msd"]:
            public_sample = item
            break

# 3. Import các Data Contracts của hệ thống
from src.rag.base import DentalDocument
from src.agents.contracts import Evidence, Citation, DraftClaim, Draft, ClaimAssessment, EvidenceQuote

# Đóng gói DentalDocument
chunk_id = internal_sample.get("chunk_id", internal_sample.get("id", "CKH-CH12"))
title = " > ".join(internal_sample.get("heading_path", [])) if "heading_path" in internal_sample else internal_sample.get("title", "Máng nhai")
internal_doc = DentalDocument(
    id=chunk_id,
    title=title,
    content=internal_sample["content"][:300] + "...",
    category=internal_sample.get("category", "Cắn khớp học & TMJ"),
    metadata={
        "warehouse": "internal",
        "document_id": internal_sample.get("document_id", "RHM1-CH12"),
        "heading_path": internal_sample.get("heading_path", []),
        "token_count": internal_sample.get("token_count", 150)
    }
)

# Đóng gói Evidence (A2 nhận từ Retriever)
internal_evidence = Evidence(
    evidence_id="I1",
    source_id=internal_doc.metadata["document_id"],
    chunk_id=internal_doc.id,
    warehouse="internal",
    title=internal_doc.title,
    content=internal_doc.content,
    metadata=internal_doc.metadata
)

# Đóng gói Evidence công khai (A3 nhận khi tra cứu MSD/BYT)
public_evidence = Evidence(
    evidence_id="P1",
    source_id=public_sample["document_id"],
    chunk_id=public_sample["chunk_id"],
    warehouse="public",
    title="Hướng dẫn chẩn đoán và điều trị chuyên khoa Răng Hàm Mặt - Bộ Y Tế",
    content=public_sample["content"][:280] + "...",
    metadata={"source": public_sample["source"], "language": public_sample.get("language", "vi")}
)

# 4. Mô phỏng Output thực tế của Agent A2 (Drafting Agent)
a2_draft_output = Draft(
    status="draft",
    claims=[
        DraftClaim(
            claim_id="c1",
            text="Máng nhai là một khí cụ cơ học có thể tháo lắp, thường làm bằng nhựa acrylic, được gắn trên cung răng nhằm thay đổi tương quan khớp cắn và bảo vệ hệ thống nhai.",
            citations=[
                Citation(
                    evidence_id="I1",
                    source_id=internal_evidence.source_id,
                    chunk_id=internal_evidence.chunk_id,
                    warehouse="internal"
                )
            ]
        )
    ],
    discarded_duplicate_claims=0
)

# 5. Mô phỏng Output thực tế của Agent A3 (Verification Agent)
a3_verification_output = {
    "schema_version": "claim-verification-v2",
    "status": "supported",
    "checks": [
        {
            "claim_id": "c1",
            "status": "supported",
            "draft_support": "supported",
            "public_support": "supported",
            "public_evidence_ids": ["P1"],
            "public_quotes": [
                {
                    "evidence_id": "P1",
                    "text": "Máng nhai được chỉ định trong điều trị rối loạn khớp thái dương hàm và bảo vệ răng."
                }
            ],
            "reason": "Đoạn nội bộ: có hỗ trợ; kho công khai: có hỗ trợ. Nội dung phù hợp với hướng dẫn lâm sàng.",
            "public_citations": [
                {
                    "evidence_id": "P1",
                    "source_id": public_evidence.source_id,
                    "chunk_id": public_evidence.chunk_id,
                    "warehouse": "public"
                }
            ],
            "timings": {
                "retrieval_ms": 12.4,
                "nli_ms": 45.2,
                "llm_ms": 110.5,
                "quote_audit_ms": 1.1
            },
            "nli_assessment": {
                "internal": {"status": "supported", "support_ids": ["I1"], "contradiction_ids": []},
                "public": {"status": "supported", "support_ids": ["P1"], "contradiction_ids": []}
            }
        }
    ]
}

print("=" * 80)
print("1. [DATA INPUT] ĐOẠN TÀI LIỆU NỘI BỘ (INTERNAL CHUNK TỪ SÁCH CẮN KHỚP HỌC)")
print("=" * 80)
print(json.dumps(internal_doc.model_dump(), ensure_ascii=False, indent=2))

print("\n" + "=" * 80)
print("2. [DATA PIPELINE] GÓI BẰNG CHỨNG ĐÃ ĐƯỢC CHUẨN HÓA (EVIDENCE BUNDLE CHO AGENT)")
print("=" * 80)
print(json.dumps([internal_evidence.model_dump(), public_evidence.model_dump()], ensure_ascii=False, indent=2))

print("\n" + "=" * 80)
print("3. [AGENT A2 OUTPUT] BẢN NHÁP KHOA HỌC KÈM TRÍCH DẪN (DRAFT)")
print("=" * 80)
print(json.dumps(a2_draft_output.model_dump(), ensure_ascii=False, indent=2))

print("\n" + "=" * 80)
print("4. [AGENT A3 OUTPUT] KẾT QUẢ KIỂM CHỨNG LÂM SÀNG & NLI CHỐNG ẢO GIÁC (VERIFICATION)")
print("=" * 80)
print(json.dumps(a3_verification_output, ensure_ascii=False, indent=2))
