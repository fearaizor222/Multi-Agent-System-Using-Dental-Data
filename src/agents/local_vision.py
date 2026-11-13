"""Experimental image findings; separate from the source-verified text answer."""
import json
from typing import Literal
from pydantic import Field
from src.agents.contracts import StrictModel
from src.agents.structured_output import parse_json


class Pathology(StrictModel):
    name: str
    location: str
    severity: Literal['mild', 'moderate', 'severe']
    visual_evidence: str


class VisionFindings(StrictModel):
    image_type: Literal['clinical_photo', 'panoramic_xray', 'periapical_xray', 'anatomical_diagram', 'other']
    anatomical_regions: list[str]
    teeth_involved: list[str]
    detected_pathologies: list[Pathology]
    preliminary_impression: str
    clinical_urgency: Literal['low', 'medium', 'high'] | None = None


PROMPT = """Phân tích hình ảnh nha khoa được cung cấp và trả DUY NHẤT một JSON theo schema.
Chỉ mô tả đặc điểm quan sát được. Không suy ra từ tên tệp, không kê thuốc hoặc quyết định điều trị.
Nếu ảnh không rõ hoặc không phải ảnh nha khoa, ghi rõ hạn chế; không bịa tổn thương.
Schema:
""" + json.dumps(VisionFindings.model_json_schema(), ensure_ascii=False)


def analyze_image(client, path, question):
    response = client.generate('A5', image_path=str(path), prompt=PROMPT + '\nYêu cầu kèm ảnh: ' + question)
    findings = VisionFindings.model_validate(parse_json(response['raw_output']))
    return {'findings': findings.model_dump(), 'model_call': response,
            'verification_status': 'not_verified_by_a3',
            'message': findings.preliminary_impression}
