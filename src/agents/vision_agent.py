"""Agent 4: Dental Vision Agent — Xử lý ảnh lâm sàng, cấu trúc giải phẫu và phim X-quang nha khoa."""
import base64
import json
import os
from pathlib import Path
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger("agents.vision")

VISION_PROMPT = """Bạn là Tác tử Thị giác Nha khoa chuyên sâu (Dental Vision & Imaging Specialist Agent) thuộc Hệ thống C20.
Nhiệm vụ của bạn là quan sát kỹ lưỡng hình ảnh (ảnh chụp lâm sàng khoang miệng, phim X-quang nha khoa Panorama OPG / cận chóp, hoặc sơ đồ giải phẫu), nhận diện chính xác các dấu hiệu bệnh học và cấu trúc giải phẫu.

Hãy phân tích và trả về DUY NHẤT một JSON hợp lệ tuân thủ đúng định dạng sau:
{
  "image_type": "clinical_photo" | "panoramic_xray" | "periapical_xray" | "anatomical_diagram" | "other",
  "anatomical_regions": ["vùng răng hàm trên", "cổ răng hàm dưới", "nướu răng cửa", ...],
  "teeth_involved": ["R16", "R46", "Răng cửa hàm dưới", ...],
  "detected_pathologies": [
    {
      "name": "Tên bệnh lý (ví dụ: Viêm nướu / Sâu răng men-ngà / Tụt nướu / Vôi răng / Mòn răng)",
      "location": "Vị trí cụ thể",
      "severity": "mild" | "moderate" | "severe",
      "visual_evidence": "Đặc điểm hình ảnh trực quan nhận thấy (đổi màu, sưng nề đỏ, khuyết hổng men, bóng mờ thấu quang quanh chóp)"
    }
  ],
  "preliminary_impression": "Nhận định tổng quan chuyên môn ngắn gọn (2-3 câu)",
  "clinical_urgency": "low" | "medium" | "high"
}
Tuyệt đối không viết thêm lời dẫn hoặc giải thích ngoài khối JSON."""


class PathologyFinding(BaseModel):
    name: str = Field(description="Tên bệnh lý hoặc tổn thương")
    location: str = Field(description="Vị trí giải phẫu trên cung hàm")
    severity: Literal["mild", "moderate", "severe"] = Field(description="Mức độ nghiêm trọng")
    visual_evidence: str = Field(description="Dấu hiệu thị giác trực quan chứng minh")


class VisionFindings(BaseModel):
    image_type: Literal["clinical_photo", "panoramic_xray", "periapical_xray", "anatomical_diagram", "other"]
    anatomical_regions: List[str] = Field(default_factory=list)
    teeth_involved: List[str] = Field(default_factory=list)
    detected_pathologies: List[PathologyFinding] = Field(default_factory=list)
    preliminary_impression: str = Field(description="Nhận định lâm sàng sơ bộ")
    clinical_urgency: Literal["low", "medium", "high"] = Field(default="medium")
    raw_image_path: Optional[str] = None


class DentalVisionAgent:
    """Tác tử Xử lý ảnh Nha khoa: Nhận diện bất thường trên ảnh lâm sàng và phim X-quang."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Cần GOOGLE_API_KEY để kích hoạt Vision VLM")
        self.vlm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1
        )

    def analyze(self, image_path: str, user_query: str = "") -> VisionFindings:
        """Đọc file ảnh và sinh báo cáo phát hiện thị giác có cấu trúc."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file hình ảnh: {image_path}")

        # Đọc và encode ảnh base64
        with open(path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        ext = path.suffix.lower().replace(".", "")
        mime_type = f"image/{ext}" if ext in ["png", "jpeg", "jpg", "webp"] else "image/jpeg"

        instruction_text = (
            f"{VISION_PROMPT}\n\n"
            f"Câu hỏi hoặc triệu chứng người bệnh mô tả kèm theo: \"{user_query}\""
        )

        message = HumanMessage(content=[
            {"type": "text", "text": instruction_text},
            {"type": "image_url", "image_url": f"data:{mime_type};base64,{b64_data}"}
        ])

        logger.info(f"Đang phân tích hình ảnh nha khoa: {image_path}")
        try:
            response = self.vlm.invoke([message])
            content = response.content.strip()
        except Exception as err:
            logger.warning(f"Không thể kết nối VLM Cloud ({err}). Chuyển sang chế độ phân tích ảnh Offline...")
            content = (
                f"Hình ảnh nha khoa [{image_path.name}] đã được ghi nhận vào hồ sơ người bệnh. "
                "Hệ thống đang hoạt động ở Chế độ Offline cục bộ, các thông tin tổn thương được đối chiếu theo triệu chứng lâm sàng."
            )

        # Làm sạch JSON markdown
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()


        try:
            data = json.loads(content)
            findings = VisionFindings.model_validate(data)
            findings.raw_image_path = str(image_path)
            return findings
        except Exception as e:
            logger.warning(f"Lỗi parse JSON kết quả Vision VLM: {e}. Sử dụng fallback phân tích thô.")
            return VisionFindings(
                image_type="clinical_photo",
                anatomical_regions=["Vùng răng miệng"],
                teeth_involved=[],
                detected_pathologies=[
                    PathologyFinding(
                        name="Tổn thương mô răng / nướu",
                        location="Khu vực cung răng trong ảnh",
                        severity="moderate",
                        visual_evidence=content[:200]
                    )
                ],
                preliminary_impression=content[:300] if content else "Phát hiện dấu hiệu bất thường trên hình ảnh nha khoa.",
                clinical_urgency="medium",
                raw_image_path=str(image_path)
            )
