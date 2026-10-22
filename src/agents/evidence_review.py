"""Kiểm tra câu trích với văn bản gốc và ghép nhận xét trên cùng mã bằng chứng."""
import re


def body_start(evidence):
    if evidence.warehouse != "public" or "heading_path" not in evidence.metadata:
        return 0
    prefix = evidence.title + "\n" + " > ".join(evidence.metadata["heading_path"]) + "\n"
    if not evidence.content.startswith(prefix):
        raise ValueError("Chunk công khai không khớp cấu trúc tiêu đề/nội dung")
    return len(prefix)


def _collapse_with_offsets(text):
    characters, offsets = [], []
    for match in re.finditer(r"\S+", text):
        if characters:
            characters.append(" ")
            offsets.append(match.start())
        for index in range(match.start(), match.end()):
            characters.append(text[index])
            offsets.append(index)
    return "".join(characters), offsets


def audit_quote(evidence, quote):
    if quote.evidence_id != evidence.evidence_id:
        raise ValueError("Câu trích có mã bằng chứng không khớp")
    # Chỉ bỏ khác biệt xuống dòng/khoảng trắng. Không sửa chữ, dấu hoặc từ phủ định.
    normalized, mapping = _collapse_with_offsets(evidence.content)
    requested, _ = _collapse_with_offsets(quote.text)
    if not requested:
        raise ValueError("Câu trích rỗng")
    minimum = body_start(evidence)
    cursor = normalized.find(requested)
    while cursor >= 0 and mapping[cursor] < minimum:
        cursor = normalized.find(requested, cursor + 1)
    if cursor < 0:
        raise ValueError("Câu trích không có trong phần nội dung gốc của " + evidence.evidence_id)
    start, end = mapping[cursor], mapping[cursor + len(requested) - 1] + 1
    # Giữ cả đoạn chứa câu trích để không bỏ điều kiện hay phủ định ở đầu đoạn.
    context_start = max(minimum, evidence.content.rfind("\n", minimum, start) + 1)
    context_end = evidence.content.find("\n", end)
    if context_end < 0:
        context_end = len(evidence.content)
    return {"evidence_id": evidence.evidence_id, "source_id": evidence.source_id, "chunk_id": evidence.chunk_id,
        "quote": evidence.content[start:end], "char_start": start, "char_end": end, "offset_basis": "chunk.content",
        "context": evidence.content[context_start:context_end], "context_start": context_start, "context_end": context_end}


def quote_contexts(assessment, evidence):
    catalog = {item.evidence_id: item for item in evidence}
    audited, contexts = [], []
    seen = set()
    for quote in assessment.public_quotes:
        if quote.evidence_id not in catalog:
            raise ValueError("Câu trích dùng mã ngoài ngữ cảnh công khai")
        item = catalog[quote.evidence_id]
        checked = audit_quote(item, quote)
        audited.append(checked)
        key = (item.evidence_id, checked["context_start"], checked["context_end"])
        if key not in seen:
            seen.add(key)
            contexts.append(item.model_copy(deep=True, update={"content": checked["context"]}))
    return audited, contexts


def reconcile_public(assessment, nli):
    selected = set(assessment.public_evidence_ids)
    support = sorted(selected & set(nli["support_ids"]))
    contradiction = sorted(selected & set(nli["contradiction_ids"]))
    if nli.get("conflicting_evidence"):
        status, ids = "insufficient", []
    elif contradiction:
        status, ids = "contradicted", contradiction
    elif assessment.public_support == "supported" and support:
        status, ids = "supported", support
    else:
        status, ids = "insufficient", []
    return status, ids
