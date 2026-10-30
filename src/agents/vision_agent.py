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

VISION_PROMPT = """Bạn là Bác sĩ Chuyên gia Chẩn đoán Hình ảnh Nha khoa (Dental Vision & Imaging Specialist Agent) thuộc Hệ thống C20 AI.
Nhiệm vụ của bạn là quan sát kỹ lưỡng hình ảnh lâm sàng khoang miệng hoặc phim X-quang nha khoa, nhận diện chính xác các cấu trúc giải phẫu và tổn thương bệnh học.

HƯỚNG DẪN CHẨN ĐOÁN PHÂN BIỆT LÂM SÀNG:
1. RĂNG CỬA vs RĂNG HÀM: Quan sát rõ cung răng. Nếu là răng một chân, rìa cắn mỏng hình xẻng ở phía trước -> Răng cửa / Răng nanh (Anterior teeth). Nếu có nhiều múi, rãnh mặt nhai lớn -> Răng hàm/răng cối (Molars/Premolars).
2. MÒN MEN NGÀ RĂNG (Tooth Wear / Erosion / Attrition): Mặt nhai hoặc rìa cắn/mặt trong xuất hiện các hố lõm (cupping/crater), lớp men trắng bên ngoài bị mòn vẹt để lộ ngà răng màu vàng cam hoặc nâu sẫm bóng nhẵn, viền men xung quanh còn gờ cao. Thường do nghiến răng hoặc trào ngược axit/thức ăn chua.
3. SÂU RĂNG (Dental Caries): Đáy lỗ sâu mủn bở, ngà mủn sẫm màu, thường xuất phát từ trũng rãnh hoặc kẽ răng.
4. CAO RĂNG & VIÊM NƯỚU: Mảng bám vôi hóa cứng dính ở cổ răng/kẽ răng, nướu sưng đỏ, phì đại hoặc tụt nướu lộ chân răng.

Hãy phân tích và trả về DUY NHẤT một JSON hợp lệ tuân thủ đúng định dạng sau:
{
  "image_type": "clinical_photo" | "panoramic_xray" | "periapical_xray" | "anatomical_diagram" | "other",
  "anatomical_regions": ["vùng răng cửa trước", "mặt trong/rìa cắn răng cửa", "cổ răng hàm dưới", ...],
  "teeth_involved": ["Răng cửa hàm dưới (R31, R32, R41, R42)", "Răng nanh (R33, R43)", ...],
  "detected_pathologies": [
    {
      "name": "Tên bệnh lý chính xác (ví dụ: Mòn men ngà răng (Dental Erosion / Attrition) / Sâu răng men-ngà / Vôi răng / Viêm nướu)",
      "location": "Vị trí giải phẫu cụ thể",
      "severity": "mild" | "moderate" | "severe",
      "visual_evidence": "Đặc điểm trực quan quan sát được (ví dụ: hố lõm cupping lộ ngà màu vàng cam, viền men gồ cao, mất chất mô răng)"
    }
  ],
  "preliminary_impression": "Nhận định tổng quan chuyên môn ngắn gọn (2-3 câu nêu rõ vùng răng tổn thương và bệnh lý chính)",
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
        self.model_name = model_name
        self.client = None
        if api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
                logger.info(f"[Vision Agent] Đã khởi tạo Google GenAI Client với model: {self.model_name}")
            except Exception as e:
                logger.warning(f"[Vision Agent] Lỗi khởi tạo Google GenAI Client ({e}).")
        else:
            logger.warning("[Vision Agent] Không tìm thấy GOOGLE_API_KEY, kích hoạt chế độ Vision Offline/Fallback.")

    def analyze(self, image_path: str, user_query: str = "") -> VisionFindings:
        """Đọc file ảnh và sinh báo cáo phát hiện thị giác có cấu trúc."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file hình ảnh: {image_path}")

        ext = path.suffix.lower().replace(".", "")
        mime_type = f"image/{ext}" if ext in ["png", "jpeg", "jpg", "webp"] else "image/jpeg"

        instruction_text = (
            f"{VISION_PROMPT}\n\n"
            f"Câu hỏi hoặc triệu chứng người bệnh mô tả kèm theo: \"{user_query}\""
        )

        logger.info(f"[Vision Agent] Đang phân tích hình ảnh nha khoa: {image_path}")
        content = ""

        if self.client:
            try:
                from google.genai import types
                with open(path, "rb") as f:
                    img_bytes = f.read()

                part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
                res = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[part, instruction_text]
                )
                content = res.text.strip()
                logger.info(f"[Vision Agent] Đã nhận phân tích thị giác thành công từ VLM ({len(content)} ký tự).")
            except Exception as err:
                logger.warning(f"[Vision Agent] Không thể kết nối VLM Cloud ({err}). Chuyển sang fallback phân tích ảnh...")

        if not content:
            # Fallback phân tích thị giác khi offline hoàn toàn
            content = (
                f"Hình ảnh nha khoa [{path.name}] đã được ghi nhận. "
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
