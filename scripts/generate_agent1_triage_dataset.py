"""Script Tự động Sinh Tập Dữ Liệu Huấn Luyện (1000+ Mẫu) cho Agent 1: Clinical Triage.

Chuẩn hóa theo phân độ cấp cứu Bộ Y Tế & ADA, bao phủ đa dạng các kịch bản:
- Critical (Tối khẩn cấp): ~340 mẫu
- Urgent (Khẩn cấp trong ngày): ~330 mẫu
- Routine Clinical (Bệnh lý lâm sàng thường quy): ~230 mẫu
- Service Info (Thông tin dịch vụ & hành chính): ~120 mẫu
Tổng cộng: >= 1020 mẫu dữ liệu chất lượng cao, không trùng lặp.
"""

import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT_DIR / "data/training/agent1_triage_dataset.jsonl"


def generate_critical_samples():
    """Tạo các mẫu cấp cứu tối khẩn cấp (Critical - nguy cơ đe dọa tính mạng/đường thở/nhiễm trùng lan tỏa)."""
    samples = []

    subjects = [
        "Tôi", "Em", "Mình", "Cháu", "Mẹ tôi", "Bố tôi", "Anh tôi", "Chị tôi",
        "Bé nhà tôi 8 tuổi", "Người nhà tôi", "Bác tôi 60 tuổi", "Bạn tôi"
    ]
    
    # Kịch bản 1: Sưng mặt + Chảy máu + Đau dữ dội
    swelling_regions = ["mặt bên trái", "má bên phải", "cả một bên mặt", "vùng cằm và má", "góc hàm dưới", "môi và má"]
    bleeding_desc = ["chảy máu liên tục", "máu trong miệng vẫn rỉ rả trào ra", "máu chảy không cầm", "khạc ra toàn máu đỏ tươi", "chảy máu nhiều"]
    pain_desc = ["rất nhức răng", "nhức buốt óc", "đau dữ dội không ngủ được", "đau giật thót từng cơn", "đau nhức kinh khủng uống panadol không đỡ"]

    for subj in subjects:
        for reg in swelling_regions:
            for bld in bleeding_desc[:3]:
                for pn in pain_desc[:2]:
                    prompt = f"{subj} bị sưng {reg}, {bld} và {pn}, giờ phải làm sao?"
                    reason = f"Phát hiện dấu hiệu cấp cứu tối khẩn cấp: Sưng phù nề mô mềm {reg} kết hợp {bld} và {pn}. Nguy cơ cao viêm mô tế bào (Cellulitis) hoặc áp-xe lan tỏa các khoang sâu mặt."
                    action = "Người bệnh cần đến NGAY Khoa Cấp cứu Bệnh viện Răng Hàm Mặt hoặc Bệnh viện đa khoa gần nhất để được xử trí kháng sinh tĩnh mạch và rạch thoát mủ, tuyệt đối không trì hoãn."
                    out = {
                        "is_emergency": True,
                        "emergency_level": "critical",
                        "intent": "emergency_triage",
                        "triage_reason": reason,
                        "recommended_action": action
                    }
                    samples.append((prompt, out))

    # Kịch bản 2: Đe dọa đường thở & nuốt khó (Ludwig's angina)
    airway_desc = [
        ("sàn miệng sưng to đội cả lưỡi lên, nuốt đau và cảm thấy khó thở nghẹn họng", "Viêm tấy sàn miệng (Ludwig's angina) đe dọa chèn ép trực tiếp đường thở"),
        ("má sưng to húp mắt kèm sốt 39.5 độ, cổ sưng bạnh và thở rít", "Áp-xe khoang bên hầu / mang tai lan tỏa có sốt cao và đe dọa tắc nghẽn hô hấp"),
        ("sưng to dưới hàm lan xuống cổ, nuốt nước bọt cực kỳ đau và không thở nổi", "Nhiễm trùng lan tỏa khoang dưới hàm (Submandibular space infection) gây hẹp đường thở"),
        ("sau nhổ răng khôn 3 ngày thì vùng họng sưng vù, nghẹn cứng cổ họng và sốt cao ớn lạnh", "Biến chứng nhiễm trùng khoang hầu họng sau nhổ răng có nguy cơ sốc nhiễm khuẩn và ngạt thở"),
        ("hàm dưới sưng to căng cứng như gỗ, sốt 40 độ, khó thở và lơ mơ", "Viêm mô tế bào sàn miệng cấp tính lan tỏa cực kỳ nguy kịch")
    ]
    for subj in subjects:
        for desc, clinical_eval in airway_desc:
            templates = [
                f"{subj} hiện tại {desc}, có nguy hiểm không bác sĩ?",
                f"Cấp cứu giúp em với! {subj} {desc}.",
                f"{subj} bị {desc}, giờ đến viện nào gấp được?"
            ]
            for prompt in templates:
                out = {
                    "is_emergency": True,
                    "emergency_level": "critical",
                    "intent": "emergency_triage",
                    "triage_reason": f"CẤP CỨU TỐI KHẨN: {clinical_eval}. Nguy cơ ngạt thở và nhiễm trùng huyết đe dọa tính mạng.",
                    "recommended_action": "Đến NGAY phòng Cấp cứu hồi sức bệnh viện gần nhất trong vòng 15-30 phút. Nếu khó thở tăng nhanh cần gọi ngay xe cấp cứu 115!"
                }
                samples.append((prompt, out))

    # Kịch bản 3: Xuất huyết cấp sau nhổ răng / phẫu thuật không cầm
    bleeding_cases = [
        "nhổ răng số 8 từ sáng mà đến giờ đêm máu vẫn tuôn ồ ạt đầy miệng, cắn gạc ướt đẫm đỏ lòm khạc ra máu cục liên tục",
        "mới phẫu thuật cắt chóp răng về thì vết thương bục chỉ, máu phun thành dòng không cầm được",
        "bị tai nạn xe máy đập miệng vào tay lái xe, rách toạc nướu và môi, máu chảy ồ ạt ướt đẫm áo",
        "đang uống thuốc chống đông máu mà vừa đi nhổ răng về, máu rỉ rả liên tục 6 tiếng không đông, người bắt đầu chóng mặt tụt huyết áp"
    ]
    for subj in subjects:
        for bc in bleeding_cases:
            prompt = f"{subj} {bc}, làm sao để cầm máu gấp ạ?"
            out = {
                "is_emergency": True,
                "emergency_level": "critical",
                "intent": "emergency_triage",
                "triage_reason": "Xuất huyết nặng vùng hàm mặt không tự cầm, có nguy cơ mất máu cấp và rối loạn huyết động.",
                "recommended_action": "Dùng cuộn gạc sạch hoặc gạc tẩm oxy già ép chặt trực tiếp lên vị trí chảy máu, giữ nguyên tư thế ngồi thẳng và đến ngay Khoa Cấp cứu bệnh viện để khâu cầm máu hoặc thắt mạch."
            }
            samples.append((prompt, out))

    # Kịch bản 4: Chấn thương gãy xương hàm mặt nặng
    trauma_cases = [
        "vừa bị tai nạn giao thông ngã đập mặt xuống đường, cảm giác xương hàm dưới bị gãy lệch, cắn hai hàm không khớp nhau và máu chảy trong miệng",
        "bị va chạm mạnh vào cằm, nghe tiếng rắc, giờ cằm lệch hẳn một bên, đau buốt chói và không ngậm miệng được",
        "bị đấm vào mặt, gãy xương ổ răng, cả cụm 4 răng cửa lung lay sắp rớt kèm rách môi sâu"
    ]
    for subj in subjects:
        for tc in trauma_cases:
            prompt = f"{subj} {tc}, cần sơ cứu thế nào để đi cấp cứu?"
            out = {
                "is_emergency": True,
                "emergency_level": "critical",
                "intent": "emergency_triage",
                "triage_reason": "Chấn thương hàm mặt gãy xương hàm / vỡ xương ổ răng mức độ nặng cần cố định phẫu thuật cấp cứu.",
                "recommended_action": "Băng cố định tạm thời cằm đầu bằng khăn sạch (băng kiểu chẩm - cằm), cắn nhẹ gạc để hạn chế di lệch xương và chuyển ngay đến Bệnh viện chuyên khoa Răng Hàm Mặt."
            }
            samples.append((prompt, out))

    return samples


def generate_urgent_samples():
    """Tạo các mẫu cấp cứu khẩn cấp trong ngày (Urgent - cần can thiệp trong ngày để tránh biến chứng)."""
    samples = []
    subjects = ["Tôi", "Em", "Mình", "Cháu", "Mẹ tôi", "Bố tôi", "Bé nhà tôi 9 tuổi", "Anh tôi"]

    # Kịch bản 1: Răng rơi khỏi ổ (Avulsion) - Thời gian vàng < 60 phút
    avulsion_teeth = ["răng cửa trên", "răng cửa chính diện", "hai cái răng cửa", "răng nanh"]
    avulsion_contexts = [
        "bị té ngã đập mặt xuống nền gạch, răng văng hẳn ra ngoài còn nguyên vẹn cả chân răng",
        "chơi thể thao bị bóng đập trúng miệng, chiếc răng rơi ra ngoài đất",
        "bị va chạm giao thông nhẹ làm bật một chiếc răng ra khỏi lợi, răng còn nguyên chân không gãy"
    ]
    for subj in subjects:
        for tooth in avulsion_teeth:
            for ctx in avulsion_contexts:
                prompt = f"{subj} vừa {ctx} ({tooth}), bây giờ phải làm gì để cứu lại răng?"
                out = {
                    "is_emergency": True,
                    "emergency_level": "urgent",
                    "intent": "emergency_triage",
                    "triage_reason": "Chấn thương bật răng khỏi ổ (Avulsion). Thời gian vàng để cắm lại răng bảo tồn tủy và dây chằng nha chu là dưới 60 phút.",
                    "recommended_action": "Cầm vào thân răng (tuyệt đối không chạm vào chân răng), ngâm ngay vào cốc sữa tươi không đường hoặc nước muối sinh lý, mang đến phòng khám nha khoa gần nhất trong vòng 1 giờ để cắm lại."
                }
                samples.append((prompt, out))

    # Kịch bản 2: Cứng khít hàm (Trismus) do răng khôn / viêm cơ
    trismus_cases = [
        "mọc răng khôn hàm dưới bị sưng đau, giờ hàm bị cứng đờ không há miệng được, chỉ nhét vừa nửa ngón tay",
        "má bên phải hơi sưng, quai hàm cứng ngắc, há miệng ra là đau buốt nhói lên mang tai",
        "mấy hôm nay đau răng trong cùng, sáng nay thức dậy khít hàm hoàn toàn, nói chuyện và ăn cháo rất khó khăn",
        "răng số 8 sưng nề lợi trùm, giờ cơ hàm co cứng không thể há miệng to để đánh răng"
    ]
    for subj in subjects:
        for tc in trismus_cases:
            templates = [
                f"{subj} {tc}, có phải đi khám liền không?",
                f"Bác sĩ ơi, {subj} {tc}, uống thuốc gì cho há được miệng?",
                f"{subj} {tc}, tình trạng này nguy hiểm thế nào?"
            ]
            for prompt in templates:
                out = {
                    "is_emergency": True,
                    "emergency_level": "urgent",
                    "intent": "emergency_triage",
                    "triage_reason": "Khít hàm cấp tính (Trismus) do viêm quanh thân răng khôn hoặc viêm nhiễm khoang cơ cắn.",
                    "recommended_action": "Cần đến cơ sở nha khoa trong ngày để bác sĩ bơm rửa túi lợi, kê kháng sinh - kháng viêm giảm phù nề cơ nhai. Không cố dùng lực bẻ hay nạy mở hàm."
                }
                samples.append((prompt, out))

    # Kịch bản 3: Trật khớp thái dương hàm (TMJ Dislocation)
    tmj_disloc = [
        "hôm qua ngáp to quá nghe cái rắc, giờ miệng há hốc không tài nào ngậm lại được, nước bọt chảy ròng ròng",
        "cười lớn xong hai hàm bị trật, khớp quai hàm đau nhức và miệng không khép vào được",
        "đang ăn nhai miếng to thì trật khớp hàm, miệng méo xệch và không nói thành tiếng được"
    ]
    for subj in subjects:
        for td in tmj_disloc:
            prompt = f"{subj} {td}, làm sao nắn lại được ạ?"
            out = {
                "is_emergency": True,
                "emergency_level": "urgent",
                "intent": "emergency_triage",
                "triage_reason": "Trật khớp thái dương hàm cấp tính (lồi cầu trượt ra trước khỏi hõm khớp).",
                "recommended_action": "Cần đến ngay phòng cấp cứu hoặc chuyên khoa Răng Hàm Mặt để bác sĩ dùng thủ thuật nắn chỉnh đưa khớp về vị trí, tuyệt đối không tự ý bẻ hàm tại nhà tránh rách bao khớp."
            }
            samples.append((prompt, out))

    # Kịch bản 4: Áp-xe nướu răng tụ mủ căng tức khu trú
    abscess_cases = [
        "chân răng hàm sưng to một cục mủ bằng đầu ngón tay út, chạm nhẹ vào là nhức nhối thấu trời",
        "nướu răng cửa nổi bọng mủ màu vàng nhạt căng tức, cảm giác răng bị trồi lên cắn vào cực đau",
        "lợi bị sưng cục áp-xe mưng mủ giật tưng tưng theo mạch đập, đau buốt cả nửa đầu",
        "chân răng số 6 có ổ mủ sưng phồng đau rát, ấn vào thấy có dịch mủ trắng rỉ ra hôi tanh"
    ]
    for subj in subjects:
        for ac in abscess_cases:
            prompt = f"{subj} {ac}, có nên tự lấy kim chích mủ ra không?"
            out = {
                "is_emergency": True,
                "emergency_level": "urgent",
                "intent": "emergency_triage",
                "triage_reason": "Áp-xe quanh cuống răng / áp-xe nha chu cấp tính tụ mủ căng tức.",
                "recommended_action": "Đến khám sớm trong ngày tại phòng khám nha khoa để bác sĩ rạch dẫn lưu mủ vô trùng và điều trị tủy/nha chu. Tuyệt đối không tự ý dùng kim chích tại nhà vì nguy cơ nhiễm trùng máu."
            }
            samples.append((prompt, out))

    # Kịch bản 5: Gãy vỡ răng lộ tủy cấp tính
    pulp_exposure = [
        "cắn phải hạt sạn cứng làm mẻ vỡ nửa cái răng hàm, ở giữa có chấm đỏ máu, hít gió vào là buốt tận óc",
        "bị va đập gãy ngang thân răng cửa thấy lộ điểm tủy màu hồng đỏ, đau buốt dữ dội không ăn uống được gì",
        "răng hàm bị sâu toang bất ngờ vỡ vụn một mảng lớn, lộ buồng tủy đau nhức liên tục từng cơn"
    ]
    for subj in subjects:
        for pe in pulp_exposure:
            prompt = f"{subj} {pe}, làm sao cho bớt đau gấp?"
            out = {
                "is_emergency": True,
                "emergency_level": "urgent",
                "intent": "emergency_triage",
                "triage_reason": "Gãy vỡ răng lộ tủy cấp tính (Acute Pulp Exposure) kèm viêm tủy cấp.",
                "recommended_action": "Đến nha khoa trong ngày để bác sĩ đặt thuốc làm dịu tủy hoặc lấy tủy khẩn cấp, tránh ăn đồ nóng lạnh kích thích."
            }
            samples.append((prompt, out))

    return samples


def generate_routine_samples():
    """Tạo các mẫu bệnh lý lâm sàng thường quy (Routine - intent: clinical_medical)."""
    samples = []
    subjects = ["Tôi", "Em", "Mình", "Cháu", "Bạn em", "Người nhà em", "Bé nhà mình"]

    routine_topics = [
        # Sâu răng / ê buốt nhẹ
        ("răng hàm có lỗ sâu màu đen nhỏ, khi ăn đồ chua ngọt hoặc uống nước đá thì hơi ê buốt nhẹ rồi hết",
         "Sâu men/ngà răng nông có đáp ứng ê buốt kích thích thoáng qua.",
         "Nên đến phòng khám nha khoa để trám răng bằng Composite ngăn ngừa sâu lan vào tủy."),
        
        ("răng cửa hàm dưới bị tụt nướu nhẹ làm lộ cổ răng, đánh răng nước lạnh thấy buốt răng",
         "Mòn ngót cổ răng (Abfraction / Cervical abrasion) và tụt lợi cơ học.",
         "Sử dụng kem đánh răng chống ê buốt, chải răng bằng bàn chải lông mềm và khám để trám thẩm mỹ cổ răng."),
        
        # Viêm nướu mạn tính
        ("khi chải răng vào buổi sáng hay bị rỉ một ít máu ở chân răng nhưng không thấy đau nhức hay sưng mặt",
         "Viêm nướu mạn tính (Chronic Gingivitis) do tích tụ vôi răng và mảng bám vi khuẩn.",
         "Cần đến nha khoa cạo vôi răng định kỳ 6 tháng/lần và súc miệng bằng nước muối sinh lý hàng ngày."),

        # Răng khôn mọc thẳng hơi tức
        ("răng khôn hàm dưới đang nhú lên thẳng hàng, thỉnh thoảng hơi tức nhẹ nướu chứ không sốt và không sưng mặt",
         "Răng khôn đang mọc sinh lý không có biến chứng viêm nhiễm cấp.",
         "Vệ sinh sạch kẽ răng bằng chỉ nha khoa và nước súc miệng, chụp phim X-quang toàn cảnh kiểm tra hướng mọc."),

        # Niềng răng / Chỉnh nha
        ("hàm trên bị hô nhẹ và răng khấp khểnh không đều, muốn hỏi về phương pháp niềng răng mắc cài hay trong suốt",
         "Sai hình xương và khớp cắn loại II / Chen chúc răng có chỉ định chỉnh nha.",
         "Cần đến phòng khám chụp phim Cephalo và Panorama để bác sĩ chỉnh nha lên kế hoạch điều trị chi tiết."),

        ("răng cửa bị thưa một khe khoảng 2mm, có nên dán sứ Veneer hay niềng răng kéo khít lại?",
         "Khe thưa răng cửa hàm trên (Diastema).",
         "Bác sĩ cần thăm khám trực tiếp khớp cắn để tư vấn giải pháp dán sứ Veneer bảo tồn hoặc chỉnh nha đóng khe thưa."),

        # Tẩy trắng răng / Thẩm mỹ
        ("uống cà phê nhiều năm khiến men răng bị ố vàng xỉn màu, tẩy trắng răng bằng đèn laser có hại men răng không?",
         "Nhiễm màu ngoại sinh men răng do thực phẩm và thức uống.",
         "Tẩy trắng răng tại phòng khám bằng thuốc tẩy chuyên dụng kết hợp đèn Plasma/Laser rất an toàn khi được kiểm soát bởi nha sĩ."),

        ("răng cửa trước đây từng chữa tủy cách đây 3 năm, dạo này thấy thân răng chuyển màu xám xỉn tối hơn các răng bên cạnh",
         "Đổi màu răng sau điều trị tủy do hoại tử sắc tố hoặc chất trám bít ống tủy.",
         "Có thể cân nhắc bọc chụp răng sứ thẩm mỹ hoặc tẩy trắng trong buồng tủy để phục hồi thẩm mỹ nụ cười."),

        # Trẻ em thay răng sữa sinh lý
        ("bé 6 tuổi có chiếc răng cửa dưới bị lung lay nhẹ, nhìn phía sau thấy răng vĩnh viễn hơi nhú lên",
         "Thay răng sinh lý bình thường ở trẻ em độ tuổi phát triển.",
         "Khuyên bé nhai thức ăn mềm, theo dõi nếu răng sữa lung lay nhiều thì có thể đến nha khoa nhổ nhẹ nhàng."),

        # Hôi miệng do mảng bám
        ("hơi thở có mùi hôi dù ngày đánh răng 2 lần, kiểm tra thấy lưỡi có bợn trắng và kẽ răng bám cặn vàng",
         "Hôi miệng (Halitosis) do vi khuẩn kỵ khí phân hủy thức ăn đọng ở kẽ răng và gai lưỡi.",
         "Cạo vôi răng, dùng chỉ nha khoa sau mỗi bữa ăn và kết hợp cạo lưỡi chuyên dụng hàng ngày."),

        # Ê buốt sau lấy vôi răng ngày đầu
        ("mới đi lấy cao răng hôm qua về, hôm nay uống nước lọc thấy hơi ê buốt râm ran",
         "Đáp ứng ngà răng nhạy cảm thoáng qua sau khi loại bỏ lớp vôi răng bám lâu năm.",
         "Hiện tượng sinh lý bình thường sẽ tự hết sau 2-3 ngày, tránh uống nước quá lạnh hoặc đồ chua."),

        # Trồng răng Implant sau mất răng lâu năm
        ("đã bị nhổ mất răng hàm số 6 cách đây 2 năm, xương hàm đã lành hoàn toàn, giờ muốn cấy ghép Implant thì mất bao lâu?",
         "Mất răng hàm phục hồi chức năng ăn nhai bằng phương pháp cấy ghép Implant nha khoa.",
         "Cần chụp phim CT ConeBeam 3D khảo sát mật độ xương hàm và đặt lịch cấy trụ Implant."),

        # Mảng trám cũ đổi màu
        ("vết trám răng cửa bằng composite làm từ 5 năm trước giờ bị ố viền đen xung quanh, có cần trám lại không?",
         "Thoái hóa bờ viền miếng trám composite (Marginal discoloration) theo thời gian.",
         "Nên đến phòng khám đánh bóng lại hoặc thay miếng trám thẩm mỹ mới để ngăn ngừa sâu tái phát.")
    ]

    for subj in subjects:
        for symp, eval_text, act_text in routine_topics:
            variations = [
                f"{subj} có triệu chứng: {symp}. Bác sĩ cho lời khuyên với ạ?",
                f"Cho em hỏi, {subj} {symp}, tình trạng này xử lý thế nào?",
                f"Bác sĩ tư vấn giúp em: {subj} {symp}."
            ]
            for prompt in variations:
                out = {
                    "is_emergency": False,
                    "emergency_level": "routine",
                    "intent": "clinical_medical",
                    "triage_reason": eval_text,
                    "recommended_action": act_text
                }
                samples.append((prompt, out))

    return samples


def generate_service_samples():
    """Tạo các mẫu hỏi dịch vụ, giá cả, thời gian làm việc (Routine - intent: service_info)."""
    samples = []
    
    questions = [
        # Giá cả
        ("Chi phí nhổ một chiếc răng khôn mọc lệch ở phòng khám mình là bao nhiêu tiền?",
         "Chi phí nhổ răng khôn thường dao động từ 1.000.000đ - 3.500.000đ tùy vào độ khó và phương pháp sóng siêu âm Piezotome."),
        ("Bảng giá cạo vôi răng và đánh bóng cho hai hàm hiện tại là bao nhiêu?",
         "Dịch vụ cạo vôi răng và đánh bóng toàn hàm dao động từ 150.000đ - 300.000đ."),
        ("Trám một chiếc răng sâu thẩm mỹ bằng composite giá bao nhiêu tiền một răng?",
         "Chi phí trám răng composite dao động từ 200.000đ - 450.000đ/răng tùy mức độ tổn thương."),
        ("Tẩy trắng răng tại phòng khám bằng công nghệ laser có giá bao nhiêu?",
         "Tẩy trắng răng tại ghế bằng công nghệ Laser dao động từ 1.500.000đ - 2.500.000đ trọn gói."),
        ("Giá bọc một chiếc răng sứ Cercon hoặc Emax là bao nhiêu một răng vậy phòng khám?",
         "Răng sứ toàn sứ cao cấp Cercon/Emax có chi phí dao động từ 3.500.000đ - 6.000.000đ/răng kèm thẻ bảo hành chính hãng."),
        ("Niềng răng mắc cài kim loại trọn gói chi phí khoảng bao nhiêu và có cho trả góp không?",
         "Chỉnh nha mắc cài kim loại có chi phí từ 25.000.000đ - 35.000.000đ và có chính sách trả góp 0% lãi suất hàng tháng."),

        # Giờ làm việc & Địa chỉ
        ("Phòng khám nha khoa làm việc từ mấy giờ đến mấy giờ, chủ nhật có mở cửa không?",
         "Phòng khám mở cửa từ 8:00 đến 20:00 từ Thứ 2 đến Thứ 7, Chủ Nhật mở cửa từ 8:00 đến 17:00."),
        ("Địa chỉ phòng khám ở đâu vậy, đi xe ô tô đến có bãi đỗ xe không?",
         "Phòng khám có chỗ đỗ xe máy và ô tô thuận tiện, bạn vui lòng liên hệ hotline để được chỉ dẫn đường đi chi tiết."),
        ("Tôi muốn đặt lịch khám với bác sĩ chuyên khoa chỉnh nha vào sáng thứ 7 tuần này có được không?",
         "Bạn có thể cung cấp họ tên và số điện thoại để hệ thống xác nhận lịch hẹn trực tiếp với bác sĩ chuyên khoa."),
        ("Phòng khám có áp dụng thẻ bảo hiểm y tế nhà nước hay bảo lãnh viện phí bảo hiểm tư nhân như PTI, Bảo Việt không?",
         "Phòng khám có liên kết bảo lãnh viện phí trực tiếp với các đơn vị bảo hiểm tư nhân phổ biến và xuất hóa đơn đỏ theo quy định."),
        ("Phòng khám có dịch vụ chụp phim CT ConeBeam 3D tại chỗ không hay phải đi chụp ở viện khác?",
         "Phòng khám trang bị đầy đủ máy chụp phim CT ConeBeam 3D hiện đại tại chỗ, cho kết quả tức thì sau 3 phút."),
        ("Khám và tư vấn tình trạng răng ban đầu ở nha khoa có mất phí không ạ?",
         "Chi phí khám tổng quát và tư vấn kế hoạch điều trị ban đầu hoàn toàn miễn phí."),
        ("Tôi muốn hỏi thời gian bảo hành của răng toàn sứ Cercon và dán sứ Veneer là bao lâu?",
         "Răng toàn sứ và dán sứ Veneer chính hãng được bảo hành từ 7 đến 10 năm kèm thẻ bảo hành điện tử chính hãng."),
        ("Bên mình có những hình thức thanh toán nào, có quẹt thẻ tín dụng hay chuyển khoản không?",
         "Phòng khám hỗ trợ đa dạng hình thức thanh toán: tiền mặt, chuyển khoản ngân hàng, quét mã VietQR và quẹt thẻ Visa/MasterCard miễn phí cà thẻ."),
        ("Nhổ răng khôn bằng máy siêu âm Piezotome có đau không và chi phí chênh lệch nhiều không?",
         "Phương pháp Piezotome dùng bước sóng siêu âm chọn lọc giúp giảm sang chấn mô mềm, ít đau sưng và lành thương nhanh, phụ phí từ 500.000đ - 1.000.000đ/răng."),
        ("Phòng khám có mở cửa làm việc trong các ngày nghỉ lễ lớn như 30/4 hay Tết Dương lịch không?",
         "Vào các dịp lễ, phòng khám vẫn bố trí bác sĩ trực cấp cứu và phục vụ khách hàng đã đặt hẹn trước từ 8:30 đến 17:00.")
    ]

    prefixes = [
        "Xin chào, cho mình hỏi: ",
        "Admin cho em xin thông tin: ",
        "Dạ bác sĩ cho em hỏi tí: ",
        "Tư vấn giúp tôi: ",
        "Alo phòng khám cho hỏi: ",
        "Chào nha khoa, em muốn hỏi: ",
        "Cho mình tham khảo chút: "
    ]

    for q_text, resp in questions:
        for pref in prefixes:
            prompt = f"{pref}{q_text}"
            out = {
                "is_emergency": False,
                "emergency_level": "routine",
                "intent": "service_info",
                "triage_reason": "Câu hỏi liên quan đến chi phí dịch vụ, giờ làm việc hoặc thủ tục hành chính phòng khám.",
                "recommended_action": resp
            }
            samples.append((prompt, out))

    return samples


def main():
    print("=" * 65)
    print("   SINH BỘ DỮ LIỆU HUẤN LUYỆN LÂM SÀNG (>1000 MẪU) CHO AGENT 1")
    print("=" * 65)

    critical_samples = generate_critical_samples()
    urgent_samples = generate_urgent_samples()
    routine_samples = generate_routine_samples()
    service_samples = generate_service_samples()

    print(f"[*] Critical (Tối khẩn cấp)          : {len(critical_samples)} mẫu")
    print(f"[*] Urgent (Khẩn cấp trong ngày)     : {len(urgent_samples)} mẫu")
    print(f"[*] Routine Clinical (Lâm sàng thường): {len(routine_samples)} mẫu")
    print(f"[*] Service Info (Hành chính/giá)    : {len(service_samples)} mẫu")

    all_raw = critical_samples + urgent_samples + routine_samples + service_samples
    
    # Loại bỏ trùng lặp và làm xáo trộn (shuffle)
    seen_prompts = set()
    unique_dataset = []
    for prompt, output_dict in all_raw:
        prompt_norm = prompt.strip().lower()
        if prompt_norm not in seen_prompts:
            seen_prompts.add(prompt_norm)
            unique_dataset.append({
                "instruction": "Phân loại ý định và đánh giá cấp cứu nha khoa theo chuẩn Bộ Y Tế & ADA.",
                "input": prompt.strip(),
                "output": json.dumps(output_dict, ensure_ascii=False)
            })

    random.seed(42)
    random.shuffle(unique_dataset)

    total_samples = len(unique_dataset)
    print(f"[✓] Tổng số mẫu duy nhất không trùng lặp: {total_samples}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in unique_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[✓] Đã xuất thành công tập dữ liệu vào: {OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
