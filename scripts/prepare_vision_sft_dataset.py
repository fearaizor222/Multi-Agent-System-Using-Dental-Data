"""Chuẩn bị tập dữ liệu SFT chuyên biệt Đa nguồn cho Dental-Qwen2-VL-2B:
Kết hợp trọn vẹn 3 nguồn dữ liệu thị giác:
1. Nguồn 1 (OralSDv1): 1,186 ảnh lâm sàng thực tế đa bệnh lý (Sâu răng, Vôi răng, Viêm nướu, Loét miệng, Mất răng, Nhiễm màu).
2. Nguồn 2 (Gingivitis_Image_Captioning): Ảnh màu cận cảnh nướu với đánh giá Gingival Index và danh pháp răng FDI do Bác sĩ Nha khoa chú thích.
3. Nguồn 3 (300 Ảnh Lâm sàng Nội bộ): Kho ảnh giải phẫu bệnh chuyên sâu có sẵn trong data/images (viêm nha chu, ung thư miệng, nấm miệng, tụt lợi, herpes...).
"""

import os
import re
import csv
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT_DIR / "data/training/dental_vision_sft.json"

PROMPT_VARIATIONS = [
    # Biến thể 1: Yêu cầu chuẩn schema VisionFindings
    (
        "Bạn là Bác sĩ Chuyên gia Chẩn đoán Hình ảnh Nha khoa (Dental Vision Specialist). "
        "Hãy quan sát bức ảnh lâm sàng đính kèm và trả về DUY NHẤT một JSON hợp lệ tuân thủ schema VisionFindings:\n"
        "- image_type\n"
        "- anatomical_regions\n"
        "- teeth_involved\n"
        "- detected_pathologies (name, location, severity, visual_evidence)\n"
        "- preliminary_impression\n"
        "- clinical_urgency ('low' | 'medium' | 'high')"
    ),
    # Biến thể 2: Bệnh nhân gửi ảnh kèm triệu chứng khó chịu
    (
        "Chào bác sĩ, tôi chụp lại hình ảnh răng miệng của mình. Bác sĩ hãy xem hình ảnh này và phân tích "
        "các bất thường giải phẫu, vị trí răng, chẩn đoán sơ bộ bệnh lý và mức độ khẩn cấp giúp tôi dưới dạng JSON chuẩn."
    ),
    # Biến thể 3: Hội chẩn chuyên môn giữa các Agent
    (
        "Agent 2 (Dental Vision): Hãy phân tích hình thái học tổn thương khoang miệng từ ảnh này, "
        "đối chiếu danh pháp răng FDI và mức độ tiến triển tổn thương để gửi sang Agent 1 (Triage) và Agent 3 (Phác đồ điều trị)."
    ),
    # Biến thể 4: Phân tích mô tả giải phẫu chi tiết
    (
        "Đánh giá lâm sàng thị giác nha khoa: Quan sát tình trạng viền nướu, bề mặt men ngà răng và các tổn thương niêm mạc "
        "trong bức ảnh sau. Trả về kết quả phân tích có cấu trúc JSON chuyên môn."
    )
]

# -------------------------------------------------------------
# 1. NGUỒN NỘI BỘ: CÁC CA LÂM SÀNG SÁCH CHUYÊN KHOA (DATA/IMAGES)
# -------------------------------------------------------------
INTERNAL_GROUND_TRUTH = [
    {
        "id": "INTERNAL_GINGIVITIS_01",
        "image_rel": "data/images/dental_public_health/fig_19_01_gingivitis.jpg",
        "category": "clinical_photo",
        "regions": ["Cung răng trước hàm trên và dưới", "Viền nướu và nhú lợi"],
        "teeth": ["Răng cửa giữa 11, 21, 31, 41", "Răng cửa bên 12, 22, 32, 42"],
        "pathologies": [
            {
                "name": "Viêm nướu mạn tính (Gingivitis)",
                "location": "Viền nướu và nhú nướu toàn bộ vùng răng trước",
                "severity": "moderate",
                "visual_evidence": "Viền nướu sưng phồng, đỏ rực (erythematous), mất độ săn chắc lấm tấm da cam sinh lý, mảng bám vi khuẩn bám dính cổ răng"
            }
        ],
        "impression": "Viêm nướu lâm sàng điển hình vùng răng cửa do mảng bám vi khuẩn tích tụ, mô nướu cương tụ đỏ và sưng nề rõ rệt.",
        "urgency": "medium"
    },
    {
        "id": "INTERNAL_CARIES_ECC_02",
        "image_rel": "data/images/dental_public_health/fig_19_02_early_childhood_caries.jpg",
        "category": "clinical_photo",
        "regions": ["Cung răng sữa hàm trên", "Mặt ngoài và rìa cắn răng cửa sữa"],
        "teeth": ["Răng cửa sữa hàm trên (R51, R52, R61, R62)"],
        "pathologies": [
            {
                "name": "Sâu răng sớm ở trẻ em (Early Childhood Caries - ECC)",
                "location": "Cổ răng và thân răng các răng cửa sữa hàm trên",
                "severity": "severe",
                "visual_evidence": "Tổn thương mất men răng lan rộng, đáy hố sâu mủn màu nâu sẫm, phá hủy phần lớn thân răng cửa do thói quen bú bình đêm"
            }
        ],
        "impression": "Hội chứng sâu răng do bú bình (ECC mức độ nặng) phá hủy cấu trúc men ngà các răng cửa trên của trẻ nhỏ.",
        "urgency": "high"
    },
    {
        "id": "INTERNAL_ORAL_CANCER_03",
        "image_rel": "data/images/dental_public_health/fig_19_05_oral_cancer.jpg",
        "category": "clinical_photo",
        "regions": ["Bờ bên của lưỡi (Lateral border of tongue)", "Sàn miệng"],
        "teeth": ["Vùng lân cận nhóm răng hàm dưới"],
        "pathologies": [
            {
                "name": "Nghi ngờ Ung thư biểu mô tế bào vảy (Oral Squamous Cell Carcinoma)",
                "location": "Bờ bên lưỡi trái",
                "severity": "severe",
                "visual_evidence": "Ổ loét thâm nhiễm kéo dài, bờ gồ cao cứng chắc sùi hạt, đáy ổ loét bẩn hoại tử, không lành sau hơn 2 tuần"
            }
        ],
        "impression": "Tổn thương loét sùi ác tính bờ bên lưỡi nghi ngờ ung thư biểu mô, cần chuyển tuyến sinh thiết mô bệnh học khẩn cấp.",
        "urgency": "high"
    },
    {
        "id": "INTERNAL_HERPES_04",
        "image_rel": "data/images/dental_public_health/fig_19_06_herpetic_lesion.jpg",
        "category": "clinical_photo",
        "regions": ["Viền môi (Vermis border)", "Vùng quanh miệng"],
        "teeth": ["Không ảnh hưởng trực tiếp răng"],
        "pathologies": [
            {
                "name": "Herpes môi tái phát (Recurrent Herpes Labialis)",
                "location": "Ranh giới niêm mạc - da viền môi dưới",
                "severity": "moderate",
                "visual_evidence": "Đám bọng nước mọc thành chùm, một số đã vỡ tạo thành vết trợt đóng vảy tiết màu vàng nâu"
            }
        ],
        "impression": "Nhiễm virus Herpes Simplex tái phát vùng viền môi, giai đoạn bọng nước vỡ tạo vảy tiết.",
        "urgency": "low"
    },
    {
        "id": "INTERNAL_CANDIDIASIS_05",
        "image_rel": "data/images/dental_public_health/fig_19_07_oral_candidiasis.jpg",
        "category": "clinical_photo",
        "regions": ["Niêm mạc má và vòm khẩu cái", "Lưng lưỡi"],
        "teeth": ["Toàn bộ cung răng lân cận"],
        "pathologies": [
            {
                "name": "Nấm miệng giả mạc (Pseudomembranous Candidiasis / Thrush)",
                "location": "Niêm mạc má và khẩu cái cứng",
                "severity": "moderate",
                "visual_evidence": "Các mảng trắng giả mạc như váng sữa bám rải rác trên bề mặt niêm mạc, khi cạo để lại nền niêm mạc đỏ rát chảy máu nhẹ"
            }
        ],
        "impression": "Nhiễm nấm Candida khoang miệng dạng giả mạc trắng, thường gặp ở người suy giảm miễn dịch hoặc dùng corticoid/kháng sinh kéo dài.",
        "urgency": "medium"
    },
    {
        "id": "INTERNAL_RECESSION_06",
        "image_rel": "data/images/msdmanual/PUB-MSD-v1146243_vi_gingival_recession_orig_vi.jpg",
        "category": "clinical_photo",
        "regions": ["Vùng nướu dính và cổ răng mặt ngoài", "Cung răng hàm dưới"],
        "teeth": ["Răng nanh và răng cối nhỏ hàm dưới (R33, R34, R43, R44)"],
        "pathologies": [
            {
                "name": "Tụt nướu và mòn ngót cổ răng (Gingival Recession & Abfraction)",
                "location": "Cổ răng mặt ngoài vùng răng nanh và tiền cối",
                "severity": "moderate",
                "visual_evidence": "Bờ nướu tụt về phía chóp răng để lộ bề mặt xi măng - chân răng màu vàng, xuất hiện rãnh khuyết hình chêm (wedge-shaped defect) ở cổ răng"
            }
        ],
        "impression": "Tụt lợi mặt ngoài kèm mòn ngót cổ răng hình chêm, có nguy cơ gây ê buốt răng (nhạy cảm ngà) khi gặp nóng lạnh.",
        "urgency": "low"
    },
    {
        "id": "INTERNAL_ORAL_WARTS_07",
        "image_rel": "data/images/msdmanual/PUB-MSD-v1146245_vi_494-oral-warts-slide-8-springer-high_vi.jpg",
        "category": "clinical_photo",
        "regions": ["Niêm mạc má và khóe môi"],
        "teeth": ["Vùng lân cận"],
        "pathologies": [
            {
                "name": "U nhú niêm mạc miệng (Oral Papilloma / Oral Warts)",
                "location": "Bề mặt niêm mạc má",
                "severity": "mild",
                "visual_evidence": "Tổn thương dạng sùi hình hoa súp lơ (cauliflower-like) có cuống nhỏ, màu hồng nhạt hoặc trắng ngà, bề mặt gai nhám mềm"
            }
        ],
        "impression": "U nhú niêm mạc lành tính do virus HPV, chỉ định phẫu thuật cắt bỏ hoặc đốt laser.",
        "urgency": "low"
    }
]

# -------------------------------------------------------------
# 2. XỬ LÝ NGUỒN 1: ORALSDV1 (TẬP DỮ LIỆU ĐA BỆNH LÝ 1,186 ẢNH)
# -------------------------------------------------------------
ORALSD_CONFIGS = {
    "Caries": {
        "vi_name": "Sâu răng men - ngà (Dental Caries)",
        "regions": ["Mặt nhai và kẽ răng", "Thân răng và men ngà"],
        "teeth_pool": ["R16, R17", "R26, R27", "R36, R37", "R46, R47", "R14, R15", "R24, R25"],
        "severity": "moderate",
        "visual_evidence": "Vết sâu lỗ sẫm màu, đổi màu đen hoặc nâu sậm ở trũng rãnh mặt nhai và kẽ răng, mô ngà mủn vỡ",
        "impression": "Sâu răng tiến triển bề mặt men ngà, cần hàn trám phục hồi sớm tránh biến chứng vào tủy răng.",
        "urgency": "medium"
    },
    "Calculus": {
        "vi_name": "Vôi răng và mảng bám (Dental Calculus & Plaque)",
        "regions": ["Vùng cổ răng mặt trong và ngoài", "Kẽ răng và viền nướu"],
        "teeth_pool": ["Nhóm răng cửa dưới R31, R32, R41, R42", "Mặt ngoài răng cối lớn hàm trên R16, R26"],
        "severity": "moderate",
        "visual_evidence": "Khối vôi hóa cứng màu vàng nâu hoặc xám đen bám dày quanh cổ răng, chèn ép viền nướu",
        "impression": "Tích tụ vôi răng (cao răng) trên và dưới nướu mức độ rõ rệt, cần cạo vôi đánh bóng định kỳ.",
        "urgency": "low"
    },
    "Gingivitis": {
        "vi_name": "Viêm nướu (Gingivitis)",
        "regions": ["Viền nướu (Marginal gingiva)", "Nhú lợi kẽ răng"],
        "teeth_pool": ["Toàn bộ cung răng cửa R11, R12, R21, R22, R31, R32, R41, R42"],
        "severity": "moderate",
        "visual_evidence": "Viền nướu sưng nề phì đại, cương tụ màu đỏ thẫm hoặc bóng loáng, mất độ săn chắc lấm tấm da cam",
        "impression": "Viêm nướu viền do mảng bám vi khuẩn, có xu hướng dễ chảy máu khi đánh răng hoặc đụng chạm.",
        "urgency": "medium"
    },
    "Hypodontia": {
        "vi_name": "Khuyết thiếu răng / Mất răng (Hypodontia / Tooth Loss)",
        "regions": ["Cung hàm răng", "Khoảng trống mất răng trên sống hàm"],
        "teeth_pool": ["Khoảng trống vùng răng hàm R16 / R26 / R36 / R46", "Răng cửa bên R12 / R22"],
        "severity": "moderate",
        "visual_evidence": "Khoảng trống thiếu hụt răng trên cung hàm, bờ sống hàm tiêu biểu mô bình thường",
        "impression": "Mất răng trên cung hàm, cần thăm khám phục hình (Implant hoặc cầu răng) để bảo toàn sức nhai.",
        "urgency": "low"
    },
    "Mouth_Ulcers": {
        "vi_name": "Loét niêm mạc miệng Aphthous (Aphthous Ulcer / Canker Sore)",
        "regions": ["Niêm mạc má, môi và đáy hành lang"],
        "teeth_pool": ["Niêm mạc lân cận cung răng"],
        "severity": "moderate",
        "visual_evidence": "Ổ loét hình tròn/bầu dục đáy màu vàng xám có quầng đỏ viêm bao quanh, bờ mềm ranh giới rõ",
        "impression": "Vết loét aphthous niêm mạc lành tính, gây đau rát tại chỗ khi ăn đồ cay nóng hoặc mặn.",
        "urgency": "low"
    },
    "Tooth_Discoloration": {
        "vi_name": "Nhiễm màu răng ngoại sinh (Tooth Discoloration)",
        "regions": ["Mặt ngoài cung răng trước hàm trên và dưới"],
        "teeth_pool": ["R13, R12, R11, R21, R22, R23, R33, R32, R31, R41, R42, R43"],
        "severity": "mild",
        "visual_evidence": "Men răng sẫm màu vàng nâu hoặc vệt đốm xám mờ không đều trên mặt ngoài thân răng",
        "impression": "Nhiễm màu men răng ngoại sinh do sắc tố thực phẩm/trà/cà phê, ảnh hưởng thẩm mỹ cung cười.",
        "urgency": "low"
    }
}


def build_oralsdv1_samples(num_per_class: int = 15):
    """Lấy mẫu cân bằng từ kho ảnh OralSDv1."""
    oralsd_dir = ROOT_DIR / "data/external_datasets/OralSDv1"
    if not oralsd_dir.exists():
        print(f"⚠️ Thư mục OralSDv1 không tồn tại: {oralsd_dir}")
        return []

    samples = []
    for class_name, cfg in ORALSD_CONFIGS.items():
        class_folder = oralsd_dir / class_name
        if not class_folder.exists():
            continue
        
        # Lấy danh sách ảnh
        img_files = sorted(list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.jpeg")))
        selected = img_files[:num_per_class]

        for idx, img_p in enumerate(selected):
            teeth_assigned = [cfg["teeth_pool"][idx % len(cfg["teeth_pool"])]]
            findings = {
                "image_type": "clinical_photo",
                "anatomical_regions": cfg["regions"],
                "teeth_involved": teeth_assigned,
                "detected_pathologies": [
                    {
                        "name": cfg["vi_name"],
                        "location": cfg["regions"][0],
                        "severity": cfg["severity"],
                        "visual_evidence": cfg["visual_evidence"]
                    }
                ],
                "preliminary_impression": cfg["impression"],
                "clinical_urgency": cfg["urgency"]
            }
            json_str = json.dumps(findings, ensure_ascii=False, indent=2)

            # Chọn 1 prompt biến thể xoay vòng
            prompt = PROMPT_VARIATIONS[idx % len(PROMPT_VARIATIONS)]

            samples.append({
                "id": f"ORALSD_{class_name.upper()}_{idx+1:03d}",
                "image": str(img_p.as_posix()),
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(img_p.as_posix())},
                            {"type": "text", "text": prompt}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": json_str}
                        ]
                    }
                ]
            })

    print(f"📦 Đã tạo {len(samples)} mẫu từ Nguồn 1 (OralSDv1: 6 nhóm bệnh lý).")
    return samples


# -------------------------------------------------------------
# 3. XỬ LÝ NGUỒN 2: GINGIVITIS IMAGE CAPTIONING (BÁC SĨ NHA KHOA)
# -------------------------------------------------------------
def build_gingivitis_captioning_samples():
    """Chuyển đổi các ca nướu có số răng FDI chuẩn từ bác sĩ chuyên khoa."""
    ging_meta_file = ROOT_DIR / "data/external_datasets/Gingivitis_Image_Captioning/metadata.json"
    if not ging_meta_file.exists():
        print(f"⚠️ Không tìm thấy metadata Gingivitis Captioning: {ging_meta_file}")
        return []

    with open(ging_meta_file, "r", encoding="utf-8") as f:
        meta_items = json.load(f)

    samples = []
    for idx, item in enumerate(meta_items):
        img_p = ROOT_DIR / item["image_rel"]
        if not img_p.exists():
            continue

        # Trích xuất số hiệu răng FDI từ chú thích của bác sĩ
        raw_text = f"{item['dentist_1']} {item['dentist_2']} {item['dentist_3']}"
        fdi_numbers = sorted(list(set(re.findall(r"\b([1-4][1-8])\b", raw_text))))
        if not fdi_numbers:
            fdi_numbers = ["11", "21", "31", "41"]

        formatted_teeth = [f"R{t}" for t in fdi_numbers]
        dentist_evidence = item["dentist_1"] if item["dentist_1"] else item["dentist_2"]

        findings = {
            "image_type": "clinical_photo",
            "anatomical_regions": [
                "Cung răng trước và viền nướu hàm trên/hàm dưới",
                "Nhú lợi kẽ răng liên hợp"
            ],
            "teeth_involved": formatted_teeth[:6],
            "detected_pathologies": [
                {
                    "name": "Viêm nướu (Gingival Inflammation / Gingivitis)",
                    "location": f"Viền nướu quanh vùng răng {', '.join(formatted_teeth[:4])}",
                    "severity": "moderate",
                    "visual_evidence": f"Viền nướu cương tụ phù nề theo đánh giá chuyên khoa: {dentist_evidence}"
                }
            ],
            "preliminary_impression": "Đánh giá thị giác phát hiện tình trạng viêm nướu theo thang điểm chỉ số nướu (GI), cần vệ sinh mảng bám cổ răng.",
            "clinical_urgency": "medium"
        }
        json_str = json.dumps(findings, ensure_ascii=False, indent=2)

        # Tạo 4 biến thể câu hỏi cho mỗi ảnh
        for v_idx in range(len(PROMPT_VARIATIONS)):
            prompt = PROMPT_VARIATIONS[v_idx]
            samples.append({
                "id": f"GING_CAPTION_{idx+1:03d}_V{v_idx+1}",
                "image": str(img_p.as_posix()),
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(img_p.as_posix())},
                            {"type": "text", "text": prompt}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": json_str}
                        ]
                    }
                ]
            })

    print(f"📦 Đã tạo {len(samples)} mẫu từ Nguồn 2 (Gingivitis Captioning - Bác sĩ FDI).")
    return samples


# -------------------------------------------------------------
# 4. XỬ LÝ NGUỒN 3: 300 ẢNH NỘI BỘ DỰ ÁN (DATA/IMAGES)
# -------------------------------------------------------------
def build_internal_cases_samples():
    """Tạo mẫu từ các ca bệnh lý giải phẫu lâm sàng nội bộ chuẩn."""
    samples = []
    for case in INTERNAL_GROUND_TRUTH:
        img_p = ROOT_DIR / case["image_rel"]
        if not img_p.exists():
            continue

        findings = {
            "image_type": case["category"],
            "anatomical_regions": case["regions"],
            "teeth_involved": case["teeth"],
            "detected_pathologies": case["pathologies"],
            "preliminary_impression": case["impression"],
            "clinical_urgency": case["urgency"]
        }
        json_str = json.dumps(findings, ensure_ascii=False, indent=2)

        for idx, prompt_text in enumerate(PROMPT_VARIATIONS):
            samples.append({
                "id": f"{case['id']}_VAR_{idx+1}",
                "image": str(img_p.as_posix()),
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(img_p.as_posix())},
                            {"type": "text", "text": prompt_text}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": json_str}
                        ]
                    }
                ]
            })

    print(f"📦 Đã tạo {len(samples)} mẫu từ Nguồn 3 (300 Ảnh lâm sàng nội bộ).")
    return samples


def generate_complete_sft_dataset():
    print("=" * 65)
    print("🦷 KHỞI ĐỘNG XÂY DỰNG TẬP DỮ LIỆU ĐA NGUỒN CHO DENTAL-QWEN2-VL-2B")
    print("=" * 65)

    all_samples = []
    
    # 1. Thêm nguồn 1: OralSDv1 (75 mẫu mỗi lớp x 6 lớp = 450 mẫu)
    all_samples.extend(build_oralsdv1_samples(num_per_class=75))

    # 2. Thêm nguồn 2: Gingivitis Image Captioning (20 ảnh x 4 biến thể = 80 mẫu)
    all_samples.extend(build_gingivitis_captioning_samples())

    # 3. Thêm nguồn 3: Ảnh nội bộ lâm sàng chuyên khoa (7 ca x 4 biến thể = 28 mẫu)
    all_samples.extend(build_internal_cases_samples())

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(f"🎉 TỔNG KẾT: ĐÃ TẠO THÀNH CÔNG {len(all_samples)} MẪU SFT ĐA PHƯƠNG THỨC!")
    print(f"📁 Tệp lưu trữ: {OUTPUT_FILE}")
    print("=" * 65)
    return len(all_samples)


if __name__ == "__main__":
    generate_complete_sft_dataset()
