"""Web Server & Giao diện Trực quan 5 Agent Nha khoa C20 (FastAPI + HTML5/Tailwind SPA)."""
import os
import shutil
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    os.system("chcp 65001 > nul 2>&1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from src.core.logger import get_logger
from src.graph.full_dental_graph import create_full_5agent_pipeline

logger = get_logger("web.app")

ROOT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT_DIR / "data/uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR = ROOT_DIR / "templates"

# Pipeline caches
_pipelines = {
    "offline": None,
    "online": None
}


def get_cached_pipeline(offline: bool):
    key = "offline" if offline else "online"
    if _pipelines[key] is None:
        logger.info(f"[Web] Đang khởi tạo pipeline cho chế độ: {key.upper()}...")
        _pipelines[key] = create_full_5agent_pipeline(offline_mode=offline)
        logger.info(f"[Web] Pipeline {key.upper()} đã sẵn sàng!")
    return _pipelines[key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo trước pipeline offline (để phục vụ ngay lập tức)
    logger.info("⚡ Đang tiền nạp hệ sinh thái 5 Agent vào bộ nhớ...")
    t0 = time.perf_counter()
    get_cached_pipeline(offline=True)
    logger.info(f"✅ Đã tiền nạp xong 5 Agent trong {time.perf_counter() - t0:.2f}s!")
    yield
    logger.info("🛑 Đang đóng server...")


app = FastAPI(
    title="Hệ thống 5 Agent Nha Khoa C20",
    description="Web UI hiển thị luồng xử lý và suy luận của 5 Agent Nha khoa đa phương thức",
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Không tìm thấy templates/index.html")
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/sample_image")
async def get_sample_image():
    sample_path = ROOT_DIR / "data/images/msdmanual/PUB-MSD-v1146243_vi_gingival_recession_orig_vi.jpg"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Chưa có ảnh mẫu")
    return FileResponse(sample_path, media_type="image/jpeg")


@app.post("/api/consult")
async def consult(
    question: str = Form(...),
    offline_mode: str = Form("true"),
    image: Optional[UploadFile] = File(None),
    sample_image_path: Optional[str] = Form(None)
):
    """Tiếp nhận câu hỏi, điều phối qua 5 Agent và trả về kết quả suy luận chi tiết."""
    is_offline = (offline_mode.lower() == "true")
    logger.info(f"[Web API] Nhận yêu cầu: \"{question[:60]}...\" | Offline: {is_offline}")

    final_image_path = None

    # 1. Xử lý ảnh tải lên nếu có
    if image and image.filename:
        safe_filename = f"{int(time.time())}_{Path(image.filename).name}"
        save_path = UPLOAD_DIR / safe_filename
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        final_image_path = str(save_path)
    elif sample_image_path:
        final_image_path = str(ROOT_DIR / sample_image_path)

    # 2. Lấy pipeline phù hợp
    pipeline = get_cached_pipeline(offline=is_offline)

    # 3. Thực thi luồng 5 Agent
    initial_state = {
        "question": question,
        "image_path": final_image_path,
        "trace": []
    }

    try:
        final_state = pipeline.invoke(initial_state)
    except Exception as e:
        logger.error(f"[Web API] Lỗi thực thi pipeline 5 Agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # 4. Trả về cấu trúc suy luận hoàn chỉnh của 5 Agent
    return {
        "status": "success",
        "question": question,
        "image_path": final_image_path,
        "offline_mode": is_offline,
        "trace": final_state.get("trace", []),
        "supervisor": {
            "intent": final_state.get("intent"),
            "is_emergency": final_state.get("is_emergency"),
            "emergency_level": final_state.get("emergency_level"),
            "triage_reason": final_state.get("triage_reason"),
            "recommended_action": final_state.get("recommended_action")
        },
        "vision": final_state.get("vision_findings"),
        "internal_rag": final_state.get("internal_evidence", []),
        "public_rag": final_state.get("public_evidence", []),
        "key_claims": final_state.get("key_claims", []),
        "final_response": final_state.get("final_response", "Không có câu trả lời.")
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🦷 HỆ THỐNG 5 AGENT NHA KHOA ĐA PHƯƠNG THỨC — WEB SERVER C20")
    print("🌐 Truy cập giao diện tại: http://localhost:8000")
    print("=" * 70 + "\n")
    uvicorn.run("app_web:app", host="0.0.0.0", port=8000, reload=False)

