"""Helper to load and bootstrap dental knowledge dataset."""

import json
import os
from typing import List
from src.rag.base import DentalDocument
from src.core.logger import get_logger

logger = get_logger("rag.sample_data")

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "raw", "dental_knowledge_sample.json"
)


def load_sample_dental_documents(file_path: str = DEFAULT_DATA_PATH) -> List[DentalDocument]:
    """Load sample dental knowledge documents from JSON."""
    if not os.path.exists(file_path):
        logger.warning(f"Không tìm thấy file mẫu tại {file_path}. Trả về danh sách rỗng.")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = [
            DentalDocument(
                id=item.get("id", f"doc_{i}"),
                title=item.get("title", ""),
                content=item.get("content", ""),
                category=item.get("category", "Chung"),
                keywords=item.get("keywords", [])
            )
            for i, item in enumerate(data)
        ]
        logger.info(f"Đã nạp {len(docs)} tài liệu nha khoa mẫu từ {os.path.basename(file_path)}.")
        return docs
    except Exception as e:
        logger.error(f"Lỗi khi đọc file mẫu nha khoa: {e}")
        return []
