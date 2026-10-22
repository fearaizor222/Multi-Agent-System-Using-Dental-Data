"""Lưu vector float32 và JSON nguồn theo từng kho, tìm kiếm bằng cosine."""
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data.public_chunks import PublicChunk
from src.rag.base import DentalDocument
from src.rag.local_embeddings import MODEL, LocalE5Embeddings, file_sha256, passage_body

ROOT = Path(__file__).resolve().parents[2]
INDEX_ROOT = ROOT / "data/vectorstore/e5_small_v1"


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_corpus(warehouse, root=ROOT):
    root = Path(root)
    if warehouse == "internal":
        path = root / "data/processed/khop_can_hoc_sections.json"
        check = json.loads((root / "outputs/internal_rhm_check/checks.json").read_text(encoding="utf-8"))
        if not check["embedding_input_ready"] or check["files"]["chunks"]["sha256"] != file_sha256(path):
            raise ValueError("Kho nội bộ chưa qua kiểm tra hoặc đã đổi sau lần kiểm tra")
        sources = [path]
        documents = []
        for item in json.loads(path.read_text(encoding="utf-8")):
            doc = DentalDocument.model_validate(item)
            if not doc.metadata.get("embedding_ready") or doc.metadata.get("doc_level") != "section":
                raise ValueError(f"Chunk nội bộ chưa sẵn sàng: {doc.id}")
            doc.metadata.update(warehouse="internal")
            documents.append(doc)
    elif warehouse == "public":
        corpus = root / "data/processed/public_v1/corpus"
        path = corpus / "chunks.jsonl"
        sources = [path, *(corpus / (name + "_documents.jsonl") for name in ["byt", "msd", "pmc"])]
        catalog = {}
        for source in sources[1:]:
            for line in source.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                catalog[item["document_id"]] = (item["title"], item["revision_id"])
        documents = []
        for line in path.read_text(encoding="utf-8").splitlines():
            chunk = PublicChunk.model_validate_json(line)
            title, revision = catalog[chunk.document_id]
            if revision != chunk.document_revision or chunk.chunking.get("tokenizer") != MODEL:
                raise ValueError(f"Sai revision hoặc tokenizer: {chunk.chunk_id}")
            metadata = chunk.model_dump(exclude={"content"})
            metadata.update(warehouse="public", source_type="public")
            documents.append(DentalDocument(id=chunk.chunk_id, title=title, content=chunk.content,
                                             category=chunk.source, metadata=metadata))
    else:
        raise ValueError("warehouse phải là internal hoặc public")
    ids = [document.id for document in documents]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Kho rỗng hoặc trùng ID chunk")
    return documents, {str(source.relative_to(root)): file_sha256(source) for source in sources}


def build_index(documents, source_files, directory, encoder, checkpoint_size=256, reuse_from=None):
    if checkpoint_size < 1:
        raise ValueError("checkpoint_size phải dương")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    inputs = ["passage: " + passage_body(doc) for doc in documents]
    lengths = encoder.token_lengths(inputs)
    for doc, actual in zip(documents, lengths):
        if not 0 < actual <= 380 or actual != doc.metadata["token_count"]:
            raise ValueError(f"Token không khớp hoặc vượt giới hạn: {doc.id} ({actual})")
    # Sắp theo độ dài để giảm padding, nhưng vector vẫn được ghi đúng hàng của ID gốc.
    order = np.argsort(-np.asarray(lengths), kind="stable")
    encoded_records = "".join(doc.model_dump_json() + "\n" for doc in documents).encode("utf-8")
    plan = {"schema_version": "dental-dense-index-v1", "count": len(documents), "encoder": encoder.identity,
            "source_files": source_files, "records_sha256": hashlib.sha256(encoded_records).hexdigest(),
            "encoding_order": "token_length_desc_stable", "max_passage_tokens": max(lengths)}
    manifest_path = directory / "manifest.json"
    records_path, vectors_path = directory / "records.jsonl", directory / "vectors.npy"
    if manifest_path.exists():
        state = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in plan.items()):
            raise ValueError(f"{directory}: model hoặc dữ liệu đã đổi; dùng thư mục index mới")
        if file_sha256(records_path) != plan["records_sha256"]:
            raise ValueError("records.jsonl không khớp checkpoint")
        if state["status"] == "complete":
            DenseIndex(directory)
            return state
        if state["status"] != "building" or not 0 <= state["completed_rows"] <= len(documents):
            raise ValueError("Checkpoint có trạng thái hoặc số hàng không hợp lệ")
        vectors = np.load(vectors_path, mmap_mode="r+", allow_pickle=False)
    else:
        if records_path.exists() or vectors_path.exists():
            raise ValueError(f"Có dữ liệu chưa rõ phiên bản trong {directory}; dùng thư mục mới")
        previous = None
        if reuse_from is not None and (Path(reuse_from) / "manifest.json").exists():
            previous = DenseIndex(reuse_from, encoder=encoder)
            if previous.manifest["encoder"] != encoder.identity:
                raise ValueError("Không tái sử dụng vector từ model khác")
        temporary = records_path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(encoded_records)
        temporary.replace(records_path)
        vectors = np.lib.format.open_memmap(vectors_path, mode="w+", dtype=np.float32,
                                            shape=(len(documents), encoder.dimension))
        state = {**plan, "status": "building", "completed_rows": 0, "elapsed_seconds": 0,
                 "runtime": encoder.runtime, "created_at": datetime.now(timezone.utc).isoformat()}
        reused = []
        if previous is not None:
            by_text = {hashlib.sha256(("passage: " + passage_body(doc)).encode()).hexdigest(): i
                       for i, doc in enumerate(previous.documents)}
            for row, text in enumerate(inputs):
                old_row = by_text.get(hashlib.sha256(text.encode()).hexdigest())
                if old_row is not None:
                    vectors[row] = previous.vectors[old_row]
                    reused.append(row)
            vectors.flush()
            state["reused_from"] = str(Path(reuse_from).resolve())
        state.update(reused_indices=reused, reused_rows=len(reused), new_embedding_rows=len(documents) - len(reused))
        write_json(manifest_path, state)
    if vectors.shape != (len(documents), encoder.dimension) or vectors.dtype != np.float32:
        raise ValueError("Sai kích thước hoặc kiểu vector trong checkpoint")
    start_time, elapsed_before = time.monotonic(), state["elapsed_seconds"]
    reused_indices = set(state.get("reused_indices", []))
    for start in range(state["completed_rows"], len(documents), checkpoint_size):
        selected = order[start:start + checkpoint_size]
        missing = np.asarray([i for i in selected if int(i) not in reused_indices], dtype=np.int64)
        if len(missing):
            values = encoder.encode_inputs([inputs[int(i)] for i in missing])
            vectors[missing] = values
        vectors.flush()
        state.update(completed_rows=start + len(selected), elapsed_seconds=elapsed_before + time.monotonic() - start_time)
        write_json(manifest_path, state)
        print(f"{directory.name}: {state['completed_rows']}/{len(documents)} chunk, {state['elapsed_seconds']:.0f}s", flush=True)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Vector chưa đầy đủ hoặc chưa chuẩn hóa")
    state.update(status="complete", vectors_sha256=file_sha256(vectors_path),
                 completed_at=datetime.now(timezone.utc).isoformat(),
                 counts_by_source=dict(Counter(str(doc.metadata.get("source", "internal")) for doc in documents)),
                 counts_by_language=dict(Counter(doc.metadata.get("language", "vi") for doc in documents)))
    write_json(manifest_path, state)
    return state


class DenseIndex:
    def __init__(self, directory, encoder=None):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["status"] != "complete" or self.manifest["completed_rows"] != self.manifest["count"]:
            raise ValueError("Index chưa chạy xong")
        for name, key in [("records.jsonl", "records_sha256"), ("vectors.npy", "vectors_sha256")]:
            if file_sha256(self.directory / name) != self.manifest[key]:
                raise ValueError(f"Index bị thay đổi: {name}")
        self.documents = [DentalDocument.model_validate_json(line) for line in
                          (self.directory / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        self.vectors = np.load(self.directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
        if self.vectors.shape != (len(self.documents), self.manifest["encoder"]["dimension"]) or len(self.documents) != self.manifest["count"]:
            raise ValueError("Số vector không khớp số chunk")
        if self.vectors.dtype != np.float32 or not np.isfinite(self.vectors).all() or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-4):
            raise ValueError("Vector trong index không hợp lệ")
        self._encoder = encoder

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = LocalE5Embeddings()
        if self._encoder.identity != self.manifest["encoder"]:
            raise ValueError("Model dùng hỏi không khớp model đã tạo index")
        return self._encoder

    def search(self, query, top_k=4, source=None, language=None):
        if top_k < 1:
            raise ValueError("top_k phải dương")
        eligible = np.asarray([i for i, doc in enumerate(self.documents)
                               if (source is None or doc.metadata.get("source") == source)
                               and (language is None or doc.metadata.get("language") == language)], dtype=np.int64)
        if not len(eligible):
            return []
        vector = np.asarray(self.encoder.embed_query(query), dtype=np.float32)
        scores = np.asarray(self.vectors[eligible] @ vector)
        best = np.argsort(-scores, kind="stable")[:top_k]
        return [self.documents[int(eligible[i])].model_copy(deep=True, update={"score": float(scores[i])}) for i in best]
