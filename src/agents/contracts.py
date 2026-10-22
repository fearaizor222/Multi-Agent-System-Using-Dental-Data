"""Dữ liệu trao đổi giữa bước truy xuất, tạo nháp và kiểm chứng."""
from typing import Literal
from copy import deepcopy

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Citation(StrictModel):
    evidence_id: str
    source_id: str
    chunk_id: str
    warehouse: Literal["internal", "public"]


class Evidence(Citation):
    title: str
    content: str
    metadata: dict


def evidence_bundle(documents, prefix, warehouse):
    evidence = []
    for i, doc in enumerate(documents, 1):
        if doc.metadata.get("warehouse") != warehouse:
            raise ValueError("Đoạn tài liệu thuộc sai kho")
        source_id = doc.metadata.get("document_id") or doc.metadata.get("parent_id")
        if not source_id:
            raise ValueError("Thiếu source_id của chunk: " + doc.id)
        evidence.append(Evidence(evidence_id=f"{prefix}{i}", source_id=source_id, chunk_id=doc.id,
            warehouse=warehouse, title=doc.title, content=doc.content, metadata=deepcopy(doc.metadata)))
    return evidence


def citation_of(evidence):
    return Citation.model_validate(evidence.model_dump(include=set(Citation.model_fields)))


class GeneratedClaim(StrictModel):
    text: str = Field(min_length=1, max_length=800)
    evidence_ids: list[str] = Field(min_length=1, max_length=3)


class GeneratedDraft(StrictModel):
    status: Literal["draft", "insufficient_evidence"]
    claims: list[GeneratedClaim] = Field(max_length=3)

    @model_validator(mode="after")
    def status_matches_claims(self):
        if (self.status == "draft") != bool(self.claims):
            raise ValueError("draft phải có nhận định; insufficient_evidence phải có claims rỗng")
        return self


class DraftClaim(StrictModel):
    claim_id: str
    text: str = Field(min_length=1, max_length=800)
    citations: list[Citation] = Field(min_length=1)


class Draft(StrictModel):
    status: Literal["draft", "insufficient_evidence"]
    claims: list[DraftClaim]
    discarded_duplicate_claims: int = 0

    @model_validator(mode="after")
    def validate_claims(self):
        if (self.status == "draft") != bool(self.claims):
            raise ValueError("Trạng thái bản nháp không khớp số nhận định")
        if len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError("Trùng ID nhận định")
        return self


class EvidenceQuote(StrictModel):
    evidence_id: str
    text: str = Field(min_length=1, max_length=1000)


class ClaimAssessment(StrictModel):
    draft_support: Literal["supported", "contradicted", "insufficient"]
    public_support: Literal["supported", "contradicted", "insufficient"]
    public_evidence_ids: list[str] = Field(max_length=3)
    public_quotes: list[EvidenceQuote] = Field(default_factory=list, max_length=3)
    # Phần diễn giải không quyết định chấp nhận bằng chứng; các trường kết luận/nguồn vẫn bắt buộc.
    reason: str = Field(default="", max_length=600)

    @model_validator(mode="after")
    def public_verdict_needs_evidence(self):
        if self.public_support != "insufficient" and not self.public_evidence_ids:
            raise ValueError("Kết luận hỗ trợ/mâu thuẫn cần chỉ ra bằng chứng công khai")
        ids = set(self.public_evidence_ids)
        quoted = {quote.evidence_id for quote in self.public_quotes}
        if not quoted <= ids:
            raise ValueError("Mã trong public_quotes phải thuộc public_evidence_ids")
        if self.public_support != "insufficient" and ids != quoted:
            raise ValueError("Mỗi mã public_evidence_ids cần một câu trích nguyên văn trong public_quotes")
        return self


def audit_citations(claim, evidence):
    catalog = {item.evidence_id: item for item in evidence}
    errors = []
    for citation in claim.citations:
        expected = catalog.get(citation.evidence_id)
        if expected is None or citation != citation_of(expected) or citation.warehouse != "internal":
            errors.append(citation.evidence_id)
    return errors
