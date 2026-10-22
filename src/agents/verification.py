"""A3: kiểm tra trích dẫn nội bộ và tìm kho công khai để đối chiếu từng nhận định."""
import json
import time

from src.agents.contracts import ClaimAssessment, audit_citations, citation_of, evidence_bundle
from src.agents.structured_output import ModelOutputError, invoke_structured
from src.agents.evidence_review import body_start, quote_contexts, reconcile_public
from src.rag.entailment import aggregate_predictions

VERIFY_SYSTEM = """Bạn đối chiếu một nhận định với hai nhóm bằng chứng được cung cấp. Chỉ trả một JSON theo schema.
draft_support: đoạn nội bộ được trích có hỗ trợ nhận định không.
public_support: các đoạn công khai có hỗ trợ, mâu thuẫn hay chưa đủ căn cứ.
supported: bằng chứng thể hiện đúng nội dung, phạm vi và điều kiện của nhận định.
contradicted: bằng chứng nói điều trái với nhận định. Chỉ thiếu thông tin thì dùng insufficient.
Không coi liên quan cùng chủ đề là đã hỗ trợ. Giữ các từ phủ định, đối tượng và điều kiện áp dụng.
Với kết luận public_support supported/contradicted, chỉ ra public_evidence_ids thực sự liên quan.
public_evidence_ids chỉ được chứa các mã P trong allowed_public_evidence_ids, tuyệt đối không lấy mã I của nguồn nội bộ.
Nếu public_support là supported hoặc contradicted, public_quotes phải có câu trích nguyên văn cho từng mã đã chọn.
Mỗi phần tử public_quotes có đúng hai trường evidence_id và text. Trích một câu ngắn cho mỗi mã.
text trong public_quotes phải lấy từ content của đúng đoạn công khai. Không lấy tiêu đề làm bằng chứng, không viết lại câu trích.
Nếu đoạn chỉ nói về chủ đề gần giống mà chưa chứng minh nhận định, chọn insufficient, public_evidence_ids=[] và public_quotes=[].
reason viết một câu ngắn bằng tiếng Việt, chỉ dựa vào bằng chứng, không thêm kiến thức ngoài tài liệu."""


class VerificationAgent:
    def __init__(self, llm, public_retriever, entailment=None):
        self.llm, self.public_retriever = llm, public_retriever
        self.entailment = entailment
        self.last_trace = []

    def run(self, question, draft, internal_evidence):
        self.last_trace = []
        checks = []
        catalog = {item.evidence_id: item for item in internal_evidence}
        for number, claim in enumerate(draft.claims, 1):
            timings = {}
            bad_ids = audit_citations(claim, internal_evidence)
            if bad_ids:
                checks.append({"claim_id": claim.claim_id, "status": "invalid_citation", "invalid_evidence_ids": bad_ids,
                    "reason": "Mã nguồn/chunk không khớp ngữ cảnh nội bộ đã cấp.", "public_evidence": []})
                continue
            # A3 tự truy xuất kho công khai, không dùng các chunk nội bộ làm nguồn đối chiếu công khai.
            start = time.perf_counter()
            hits = self.public_retriever.retrieve(question + "\n" + claim.text, top_k=2)
            timings["retrieval_ms"] = (time.perf_counter() - start) * 1000
            public = evidence_bundle(hits, "P", "public")
            cited_internal = [catalog[citation.evidence_id] for citation in claim.citations]
            if not public:
                # Không có nguồn công khai thì không giao cho LLM tự suy ra có hỗ trợ.
                internal_nli = None
                draft_support = "insufficient"
                if self.entailment is not None:
                    start = time.perf_counter()
                    internal_nli = self.entailment.assess(claim.text, cited_internal)
                    draft_support = internal_nli["status"]
                    timings["nli_ms"] = (time.perf_counter() - start) * 1000
                timings["llm_ms"] = 0
                checks.append({"claim_id": claim.claim_id,
                    "status": "contradicted" if draft_support == "contradicted" else "insufficient",
                    "draft_support": draft_support, "public_support": "insufficient", "public_evidence_ids": [],
                    "public_quotes": [], "public_quote_checks": [], "public_citations": [], "public_evidence": [],
                    "reason": "Không có đoạn công khai để đối chiếu.", "decision_origin": "no_public_evidence", "llm_not_called": True,
                    "nli_assessment": {"internal": internal_nli, "public": aggregate_predictions([])}, "timings": timings})
                continue
            user = json.dumps({"question": question, "claim": claim.text,
                "internal_evidence": [item.model_dump(include={"evidence_id", "content"}) for item in cited_internal],
                "public_evidence": [{"evidence_id": item.evidence_id, "content": item.content[body_start(item):]} for item in public],
                "allowed_public_evidence_ids": [item.evidence_id for item in public]}, ensure_ascii=False)
            allowed = {item.evidence_id: item for item in public}
            def validate(result):
                if any(key not in allowed for key in result.public_evidence_ids):
                    raise ValueError("public_evidence_ids chỉ được thuộc: " + (", ".join(allowed) or "[]") + "; không được dùng mã I")
                quote_contexts(result, public)
            start = time.perf_counter()
            assessment = invoke_structured(self.llm, ClaimAssessment, VERIFY_SYSTEM, user, self.last_trace, validate=validate)
            timings["llm_ms"] = (time.perf_counter() - start) * 1000
            if any(key not in allowed for key in assessment.public_evidence_ids):
                raise ModelOutputError("Kiểm chứng dùng mã bằng chứng không có trong ngữ cảnh")
            extra = {}
            effective = assessment.model_dump()
            if not effective["reason"]:
                effective["reason"] = "Model chưa nêu giải thích."
            start = time.perf_counter()
            audited_quotes, quoted_context = quote_contexts(assessment, public)
            timings["quote_audit_ms"] = (time.perf_counter() - start) * 1000
            citation_ids = list(dict.fromkeys(assessment.public_evidence_ids))
            if self.entailment is not None:
                start = time.perf_counter()
                internal_nli = self.entailment.assess(claim.text, cited_internal)
                # NLI chỉ xét ngữ cảnh của câu trích do LLM chọn, không cấp điểm từ đoạn khác.
                public_nli = self.entailment.assess(claim.text, quoted_context)
                timings["nli_ms"] = (time.perf_counter() - start) * 1000
                def reconcile(llm_status, nli_status):
                    if nli_status == "contradicted":
                        return "contradicted"
                    return "supported" if llm_status == nli_status == "supported" else "insufficient"
                draft_support = reconcile(assessment.draft_support, internal_nli["status"])
                public_support, citation_ids = reconcile_public(assessment, public_nli)
                disagreement = assessment.draft_support != internal_nli["status"] or assessment.public_support != public_nli["status"]
                disagreement = disagreement or public_support != assessment.public_support or set(citation_ids) != set(assessment.public_evidence_ids)
                labels = {"supported": "có hỗ trợ", "contradicted": "mâu thuẫn", "insufficient": "chưa đủ bằng chứng"}
                reason = "Đoạn nội bộ: " + labels[draft_support] + "; kho công khai: " + labels[public_support] + "."
                if disagreement:
                    reason += " Kết quả LLM và NLI khác nhau."
                effective.update(draft_support=draft_support, public_support=public_support,
                                 public_evidence_ids=citation_ids, reason=reason,
                                 public_quotes=[quote.model_dump() for quote in assessment.public_quotes if quote.evidence_id in citation_ids])
                extra = {"llm_assessment": assessment.model_dump(), "nli_assessment": {"internal": internal_nli, "public": public_nli},
                         "model_disagreement": disagreement}
            if effective["draft_support"] == "contradicted" or effective["public_support"] == "contradicted":
                status = "contradicted"
            elif effective["draft_support"] == effective["public_support"] == "supported":
                status = "supported"
            else:
                status = "insufficient"
            checks.append({"claim_id": claim.claim_id, "status": status, **effective, **extra,
                "llm_reason_missing": not bool(assessment.reason),
                "public_quote_checks": audited_quotes, "timings": timings,
                "public_citations": [citation_of(allowed[key]).model_dump() for key in citation_ids],
                "public_evidence": [item.model_dump() for item in public]})
        status = "supported" if checks and all(item["status"] == "supported" for item in checks) else "needs_review"
        return {"schema_version": "claim-verification-v2", "status": status, "checks": checks}
