"""Thống kê lượt chạy và rà trích dẫn; không chấm tính đúng y khoa của câu trả lời."""
from collections import Counter

from src.agents.contracts import Draft, Evidence, audit_citations, independent_public_sources
from src.agents.evidence_review import body_start


def _original_matches(evidence, originals):
    original = originals.get(evidence.chunk_id)
    if original is None:
        return False
    source_id = original.metadata.get("document_id") or original.metadata.get("parent_id")
    metadata = {key: value for key, value in evidence.metadata.items() if key != "retrieval"}
    expected = {key: value for key, value in original.metadata.items() if key != "retrieval"}
    return (evidence.source_id == source_id and evidence.content == original.content
            and evidence.title == original.title and evidence.warehouse == original.metadata.get("warehouse")
            and metadata == expected)


def review_run(state, internal_records, public_records):
    issues, internal, claims = [], [], []
    primary = []
    try:
        internal = [Evidence.model_validate(item) for item in state.get("internal_evidence", [])]
        primary = [Evidence.model_validate(item) for item in state.get("draft_evidence", state.get("internal_evidence", []))]
        if state.get("draft"):
            claims = Draft.model_validate(state["draft"]).claims
    except ValueError as exc:
        issues.append("invalid_saved_schema: " + str(exc))
    for evidence in internal:
        if not _original_matches(evidence, internal_records):
            issues.append("internal_source_mismatch: " + evidence.evidence_id)
    origin = state.get("draft_origin", "internal")
    if origin not in ("internal", "public") or any(e.warehouse != origin for e in primary):
        issues.append("draft_warehouse_mismatch")
    for evidence in primary:
        originals = public_records if evidence.warehouse == "public" else internal_records
        if not _original_matches(evidence, originals):
            issues.append("draft_source_mismatch: " + evidence.evidence_id)
    for claim in claims:
        for evidence_id in audit_citations(claim, primary):
            issues.append("invalid_internal_citation: " + claim.claim_id + ":" + evidence_id)
    quotes, checks = 0, state.get("verification", {}).get("checks", [])
    for check in checks:
        public = {item["evidence_id"]: Evidence.model_validate(item) for item in check.get("public_evidence", [])}
        for evidence in public.values():
            if not _original_matches(evidence, public_records):
                issues.append("public_source_mismatch: " + evidence.evidence_id)
        for quote in check.get("public_quote_checks", []):
            quotes += 1
            evidence = public.get(quote.get("evidence_id"))
            start, end = quote.get("char_start"), quote.get("char_end")
            valid = (evidence is not None and type(start) is int and type(end) is int
                     and 0 <= start < end <= len(evidence.content)
                     and quote.get("chunk_id") == evidence.chunk_id and quote.get("source_id") == evidence.source_id
                     and evidence.content[start:end] == quote.get("quote"))
            if valid:
                try:
                    valid = start >= body_start(evidence)
                except ValueError:
                    valid = False
            if not valid:
                issues.append("invalid_public_quote: " + check["claim_id"])
        for citation in check.get("public_citations", []):
            evidence = public.get(citation.get("evidence_id"))
            if (evidence is None or citation.get("chunk_id") != evidence.chunk_id
                    or citation.get("source_id") != evidence.source_id or citation.get("warehouse") != "public"):
                issues.append("invalid_public_citation: " + check["claim_id"])
        if origin == "public" and check.get("status") == "supported":
            claim = next((c for c in claims if c.claim_id == check["claim_id"]), None)
            primary_ids = {c.evidence_id for c in claim.citations} if claim else set()
            selected = [public[c["evidence_id"]] for c in check.get("public_citations", []) if c["evidence_id"] in public]
            if (check.get("verification_policy") != "public_cross_source_v1"
                    or not independent_public_sources([e for e in primary if e.evidence_id in primary_ids], selected)):
                issues.append("public_sources_not_independent: " + check["claim_id"])
        if check.get("status") == "invalid_citation":
            issues.append("verifier_flagged_invalid_citation: " + check["claim_id"])
    calls = [call for stage in state.get("model_calls", []) for call in stage.get("calls", [])]
    verification_reached = any(step["step"] == "a3_verify" for step in state.get("trace", []))
    unchecked = sum(check.get("status") == "not_checked" for check in checks)
    return {"status": state.get("status", "error"), "draft_claims": len(claims), "checked_claims": len(checks) - unchecked,
        "not_checked_claims": unchecked,
        "verification_reached": verification_reached,
        "verification_completed": verification_reached and bool(state.get("verification")),
        "claim_statuses": dict(Counter(check["status"] for check in checks)), "audited_quote_count": quotes,
        "internal_citation_count": sum(c.warehouse == "internal" for claim in claims for c in claim.citations),
        "public_draft_citation_count": sum(c.warehouse == "public" for claim in claims for c in claim.citations),
        "citation_integrity_issues": issues, "citation_integrity_ok": not issues,
        "model_attempts": len(calls), "invalid_model_attempts": sum(not call.get("valid", False) for call in calls),
        "stage_ms": {step["step"]: step["milliseconds"] for step in state.get("trace", [])},
        "scope": "Kiểm tra tính toàn vẹn nguồn và câu trích; không đánh giá hỗ trợ ngữ nghĩa hoặc độ đúng y khoa."}
