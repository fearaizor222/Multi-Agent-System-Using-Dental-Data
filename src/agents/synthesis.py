"""A4 v1: ghép nguyên nhận định đã được A3 hỗ trợ; không gọi LLM."""
import math
from typing import Literal
from urllib.parse import urlsplit,unquote

from pydantic import Field,model_validator

from src.agents.contracts import StrictModel,Draft,Evidence,Citation,ClaimAssessment,audit_citations,citation_of,independent_public_sources
from src.agents.evidence_review import quote_contexts
from src.rag.entailment import aggregate_predictions


class AnswerSource(StrictModel):
    reference_id:str
    warehouse:Literal["internal","public"]
    source_id:str
    chunk_id:str
    chunk_ids:list[str]=Field(default_factory=list)
    title:str
    pages:list[int]=Field(default_factory=list)
    url:str|None=None


class AnswerClaim(StrictModel):
    claim_id:str
    text:str=Field(min_length=1)
    reference_ids:list[str]=Field(min_length=1)


class FinalAnswer(StrictModel):
    status:Literal["answered","partial","insufficient_evidence","conflicting_evidence","needs_review"]
    message:str=Field(min_length=1)
    claims:list[AnswerClaim]=Field(default_factory=list)
    sources:list[AnswerSource]=Field(default_factory=list)
    omitted_claims:list[dict]=Field(default_factory=list)
    audit_issues:list[str]=Field(default_factory=list)
    strategy:Literal["verified_claims_v1","verified_claims_qwen3_v1","verified_claims_qwen25_adapter_v1"]="verified_claims_v1"
    model_used:bool=False
    evidence_basis:Literal["internal_public","public_cross_source"]|None=None

    @model_validator(mode="after")
    def consistent_answer(self):
        answer=self.status in ("answered","partial")
        if answer!=bool(self.claims) or (not answer and self.sources):
            raise ValueError("Trạng thái A4 không khớp nhận định/nguồn")
        ids=[s.reference_id for s in self.sources]
        if len(ids)!=len(set(ids)) or any(set(c.reference_ids)-set(ids) for c in self.claims):
            raise ValueError("Tham chiếu A4 không hợp lệ")
        return self


MESSAGES={
    "insufficient_evidence":"Hiện chưa đủ bằng chứng từ các tài liệu đã tra để trả lời câu hỏi này.",
    "conflicting_evidence":"Có nội dung mâu thuẫn hoặc bằng chứng chưa thống nhất. Tôi chưa thể đưa ra kết luận từ các tài liệu đã tra.",
    "needs_review":"Thông tin nguồn hoặc kết quả đối chiếu cần được kiểm tra lại trước khi trả lời."
}


def checked_nli(saved,allowed_ids):
    """Tính lại nhãn từ các score đã lưu, không chỉ tin cờ supported."""
    if not isinstance(saved,dict) or not saved.get("predictions"):
        raise ValueError("missing_nli_predictions")
    for row in saved["predictions"]:
        scores=row["scores"]
        if row["evidence_id"] not in allowed_ids or row.get("truncated") is not False:
            raise ValueError("invalid_nli_context")
        if set(scores)!={"entailment","neutral","contradiction"}:
            raise ValueError("invalid_nli_scores")
        if (any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in scores.values())
                or not math.isclose(sum(scores.values()),1.,abs_tol=1e-4)):
            raise ValueError("invalid_nli_scores")
    current=aggregate_predictions(saved["predictions"],saved["threshold"])
    for key in ("status","support_ids","contradiction_ids","conflicting_evidence"):
        if current[key]!=saved[key]:
            raise ValueError("nli_summary_mismatch")
    if current["status"]!="supported" or current["conflicting_evidence"]:
        raise ValueError("nli_not_supported")
    return set(current["support_ids"])


def source_of(evidence,reference_id):
    pages=sorted({s["pdf_page"] for s in evidence.metadata.get("source_spans",[])
                  if isinstance(s,dict) and type(s.get("pdf_page")) is int and s["pdf_page"]>0})
    url=None
    for key in ("source_url","url","canonical_url"):
        value=evidence.metadata.get(key)
        if isinstance(value,str):
            parsed=urlsplit(value)
            if parsed.scheme in ("http","https") and parsed.netloc and not any(c.isspace() for c in value):
                url=value
                break
    return AnswerSource(reference_id=reference_id,warehouse=evidence.warehouse,source_id=evidence.source_id,
                        chunk_id=evidence.chunk_id,chunk_ids=[evidence.chunk_id],title=evidence.title or evidence.source_id,pages=pages,url=url)


class SynthesisAgent:
    def run(self,knowledge):
        try:
            return self._run(knowledge).model_dump()
        except (ValueError,TypeError,KeyError,AttributeError,IndexError,OverflowError) as exc:
            return FinalAnswer(status="needs_review",message=MESSAGES["needs_review"],
                               audit_issues=[type(exc).__name__+": "+str(exc)]).model_dump()

    def _run(self,knowledge):
        if not knowledge.get("draft"):
            return FinalAnswer(status="insufficient_evidence",message=MESSAGES["insufficient_evidence"])
        draft=Draft.model_validate(knowledge["draft"])
        if not draft.claims:
            return FinalAnswer(status="insufficient_evidence",message=MESSAGES["insufficient_evidence"])
        review=knowledge.get("review",{})
        if (review.get("citation_integrity_ok") is not True or review.get("citation_integrity_issues")!=[]
                or review.get("verification_completed") is not True):
            raise ValueError("source_review_missing_or_failed")
        internal=[Evidence.model_validate(e) for e in knowledge.get("draft_evidence",knowledge.get("internal_evidence",[]))]
        origin=knowledge.get("draft_origin","internal")
        public_draft=origin=="public"
        if origin not in ("internal","public") or (public_draft and knowledge.get("repair_attempted") is not True):
            raise ValueError("unauthorized_draft_origin")
        catalog={e.evidence_id:e for e in internal}
        if len(catalog)!=len(internal) or any(e.warehouse!=origin for e in internal):
            raise ValueError("invalid_internal_catalog")
        checks=knowledge.get("verification",{}).get("checks",[])
        check_ids=[c["claim_id"] for c in checks]
        claim_ids={c.claim_id for c in draft.claims}
        if len(check_ids)!=len(set(check_ids)) or set(check_ids)-claim_ids:
            raise ValueError("unknown_or_duplicate_claim_check")
        by_id={c["claim_id"]:c for c in checks}
        omitted=[]
        if any(c.get("status")=="contradicted" or any(
                v.get("status")=="contradicted" or v.get("conflicting_evidence") is True
                for v in c.get("nli_assessment",{}).values() if isinstance(v,dict)) for c in checks):
            return FinalAnswer(status="conflicting_evidence",message=MESSAGES["conflicting_evidence"],
                omitted_claims=[{"claim_id":c.claim_id,"reason":"contradiction_in_verification"} for c in draft.claims])
        accepted=[]
        for claim in draft.claims:
            check=by_id.get(claim.claim_id,{})
            if check.get("status")!="supported":
                omitted.append({"claim_id":claim.claim_id,"reason":check.get("status","not_checked")})
                continue
            if audit_citations(claim,internal):
                raise ValueError("invalid_internal_citation: "+claim.claim_id)
            assessment=ClaimAssessment.model_validate({key:check[key] for key in ClaimAssessment.model_fields if key in check})
            if assessment.draft_support!="supported" or assessment.public_support!="supported":
                raise ValueError("supported_check_has_inconsistent_verdict")
            public=[Evidence.model_validate(e) for e in check.get("public_evidence",[])]
            public_catalog={e.evidence_id:e for e in public}
            if len(public_catalog)!=len(public) or any(e.warehouse!="public" for e in public):
                raise ValueError("invalid_public_catalog")
            public_ids=set(assessment.public_evidence_ids)
            if not public_ids or not public_ids<=set(public_catalog):
                raise ValueError("missing_public_source")
            quote_contexts(assessment,public)
            citations=[Citation.model_validate(c) for c in check.get("public_citations",[])]
            if len(citations)!=len(public_ids) or {c.evidence_id for c in citations}!=public_ids:
                raise ValueError("public_citation_set_mismatch")
            if any(c!=citation_of(public_catalog[c.evidence_id]) for c in citations):
                raise ValueError("public_citation_source_mismatch")
            nli=check.get("nli_assessment",{})
            cited_ids={c.evidence_id for c in claim.citations}
            if public_draft and (check.get("verification_policy")!="public_cross_source_v1"
                    or not independent_public_sources([catalog[i] for i in cited_ids],
                                                       [public_catalog[i] for i in public_ids])):
                raise ValueError("public_sources_not_independent")
            supporting_internal=checked_nli(nli.get("draft" if public_draft else "internal"),cited_ids)
            # A3 có thể đã bỏ nguồn neutral khỏi citations nhưng vẫn giữ prediction của nó.
            supporting_public=checked_nli(nli.get("public"),set(public_catalog))
            if not public_ids<=supporting_public or not supporting_internal:
                raise ValueError("nli_does_not_support_cited_sources")
            references=[catalog[c.evidence_id] for c in claim.citations if c.evidence_id in supporting_internal]
            references.extend(public_catalog[i] for i in assessment.public_evidence_ids)
            accepted.append((claim,references))
        if not accepted:
            return FinalAnswer(status="insufficient_evidence",message=MESSAGES["insufficient_evidence"],omitted_claims=omitted)
        sources=[];source_ids={};claims=[];seen=set()
        for claim,references in accepted:
            fingerprint=" ".join(claim.text.casefold().split())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            ids=[]
            for evidence in references:
                key=(evidence.warehouse,evidence.source_id)
                if key not in source_ids:
                    source_ids[key]=str(len(sources)+1)
                    sources.append(source_of(evidence,source_ids[key]))
                else:
                    saved=sources[int(source_ids[key])-1]
                    if evidence.chunk_id not in saved.chunk_ids:
                        saved.chunk_ids.append(evidence.chunk_id)
                        saved.pages=sorted(set(saved.pages)|set(source_of(evidence,saved.reference_id).pages))
                if source_ids[key] not in ids:
                    ids.append(source_ids[key])
            claims.append(AnswerClaim(claim_id=claim.claim_id,text=claim.text,reference_ids=ids))
        groups=[]
        for claim in claims:
            if groups and groups[-1][1]==claim.reference_ids:
                groups[-1][0]+=" "+claim.text
            else:
                groups.append([claim.text,claim.reference_ids])
        lines=[" ".join(text+" ["+", ".join(ids)+"]" for text,ids in groups)]
        if omitted:
            lines.append("Một số nội dung khác chưa đủ bằng chứng nên chưa được đưa vào câu trả lời.")
        lines.extend(["","Nguồn công khai:" if public_draft else "Nguồn tham khảo:"])
        for source in sources:
            location=" (trang "+", ".join(map(str,source.pages))+")" if source.pages else ""
            url=unquote(source.url) if source.url else None
            if url and any(c.isspace() or ord(c)<32 for c in url):
                url=source.url
            lines.append("["+source.reference_id+"] "+source.title+location+(" — "+url if url else ""))
        return FinalAnswer(status="partial" if omitted else "answered",message="\n".join(lines),claims=claims,
                           sources=sources,omitted_claims=omitted,
                           evidence_basis="public_cross_source" if public_draft else "internal_public")
