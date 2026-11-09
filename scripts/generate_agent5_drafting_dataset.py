"""Script Tự động Sinh Tập Dữ Liệu SFT Huấn Luyện (500+ Mẫu) cho Agent 5: Drafting & Synthesis Agent.

Mục tiêu huấn luyện:
1. Đọc tổng hợp: Câu hỏi người bệnh + Bằng chứng nội bộ [I1] + Bằng chứng Bộ Y Tế [P1] + Nhận định thị giác (nếu có).
2. Xuất DUY NHẤT một JSON hợp lệ tuân thủ schema DraftResponse:
   - key_claims: Danh sách nhận định cốt lõi kèm mã trích dẫn ([I1], [P1]).
   - draft_response: Lời tư vấn y khoa hoàn chỉnh, ân cần, chuẩn mực tiếng Việt, không bịa đặt, có hướng dẫn chăm sóc và khuyến nghị đi khám.
"""

import json
import random
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT_DIR / "data/training/agent5_drafting_sft.jsonl"

DRAFTING_SYSTEM_PROMPT = """Bạn là Trợ lý Thông tin Nha khoa C20 AI (Agent 5: Drafting & Consultation Synthesis Agent).
Nhiệm vụ: Căn cứ vào các ý kiến hội chẩn chuyên khoa và tài liệu y khoa được cung cấp, hãy trình bày bài hướng dẫn và tư vấn thông tin y tế chi tiết, chuẩn xác gửi tới người hỏi.

QUY TẮC BẮT BUỘC:
1. TUYỆT ĐỐI CHỈ DÙNG TIẾNG VIỆT CHUẨN XÁC, không chêm tiếng nước ngoài.
2. BÁM SÁT TÀI LIỆU VÀ Ý KIẾN HỘI CHẨN ĐƯỢC CẤP: Tuyệt đối không tự bịa đặt hoặc suy diễn các tác dụng phụ, biến chứng hay thuốc lạ không có trong tài liệu (như đau đầu, sưng mắt, chóng mặt, tiêu chảy, kem chống nắng...).
3. Trình bày rõ ràng theo từng phần:
   - Các phương án xử lý/điều trị theo phác đồ (ví dụ: tái khoáng hóa bằng Fluoride ở giai đoạn sớm vs hàn trám phục hồi khi đã có lỗ sâu).
   - Biện pháp vệ sinh, chăm sóc và dự phòng mảng bám tại nhà.
   - Lời khuyên đến phòng khám Răng Hàm Mặt để được bác sĩ thăm khám và thực hiện kỹ thuật.
4. Chỉ trả về DUY NHẤT một JSON hợp lệ tuân thủ đúng schema:
{
  "key_claims": [
    {"claim_id": "c1", "text": "...", "citations": ["I1", "P1"]}
  ],
  "draft_response": "Nội dung bài tư vấn hoàn chỉnh..."
}"""

# Danh sách các chủ đề bệnh lý nha khoa chuyên sâu
CLINICAL_TOPICS = [
    {
        "topic": "Sâu răng & Tái khoáng hóa Men răng",
        "keywords": ["sâu răng", "ngà nông", "buốt khi ăn đồ ngọt", "tái khoáng hóa", "fluoride", "trám răng"],
        "questions": [
            "Răng hàm số 6 của tôi khi ăn đồ ngọt hay uống nước lạnh thì bị ê buốt nhẹ, nhìn kỹ thấy có vết nâu đen ở kẽ răng. Tình trạng này là gì và điều trị ra sao?",
            "Tôi phát hiện mặt nhai răng hàm có vệt đen lấm chấm, hơi ê buốt thoáng qua khi chải răng. Tôi có cần đi trám ngay không?",
            "Con tôi 7 tuổi bị chấm đen ở răng hàm sữa, không đau nhức dữ dội nhưng ăn cơm hay giắt thức ăn. Xin bác sĩ tư vấn phương án xử lý."
        ],
        "i_title": "Cơ chế Khử khoáng và Tái khoáng Men Ngà Răng",
        "i_content": "Quá trình sâu răng là sự mất cân bằng giữa chu trình khử khoáng do acid vi khuẩn Streptococcus mutans chuyển hóa đường và chu trình tái khoáng tự nhiên từ nước bọt. Ở giai đoạn sớm (tổn thương đốm trắng hoặc sâu men nông), ion Fluoride, Canxi và Phosphat có khả năng tái khoáng hóa cấu trúc tinh thể Hydroxyapatite, phục hồi bề mặt men răng mà chưa cần can thiệp khoan mài xâm lấn.",
        "p_title": "Quy trình Kỹ thuật Hàn Trám Răng Sâu Ngà - Hướng dẫn Chuyên môn Bộ Y Tế",
        "p_content": "Chỉ định điều trị sâu ngà nông và ngà sâu chưa tổn thương tủy: Làm sạch ngà mủn hoại tử bằng mũi khoan thích hợp, bảo tồn tối đa mô răng lành. Cách ly răng, xoi mòn men ngà bằng acid phosphoric 37%, bôi keo dán dính nha khoa (Bonding) và trám bít phục hồi xoang sâu bằng Composite quang trùng hợp hoặc Glass Ionomer Cement (GIC). Đánh bóng miếng trám để tránh đọng mảng bám.",
        "claims": [
            {"claim_id": "c1", "text": "Dấu hiệu vết nâu đen kèm ê buốt thoáng qua khi ăn đồ ngọt/lạnh là biểu hiện điển hình của sâu men hoặc sâu ngà nông [P1], do mất cân bằng giữa chu trình khử khoáng và tái khoáng men răng [I1].", "citations": ["I1", "P1"]},
            {"claim_id": "c2", "text": "Nếu tổn thương còn ở giai đoạn sớm, có thể áp dụng liệu pháp tái khoáng bằng Fluoride [I1]; khi đã tạo lỗ khuyết hổng mô ngà, cần khoan sạch ngà mủn và trám phục hồi bằng Composite hoặc GIC theo hướng dẫn Bộ Y Tế [P1].", "citations": ["I1", "P1"]}
        ],
        "treatment_plan": "Hàn trám thẩm mỹ phục hồi xoang sâu ngà bằng vật liệu Composite hoặc GIC; bôi gel Fluoride tái khoáng hóa men răng",
        "home_care": "Chải răng 2 lần/ngày bằng kem đánh răng chứa Fluoride nồng độ 1450ppm, dùng chỉ nha khoa làm sạch kẽ răng hàm sau mỗi bữa ăn, hạn chế ăn vặt đồ ngọt vào ban đêm."
    },
    {
        "topic": "Viêm tủy răng Cấp tính & Điều trị Nội nha",
        "keywords": ["viêm tủy", "đau giật thót từng cơn", "đau buốt nửa đầu", "tủy hoại tử", "chữa tủy"],
        "questions": [
            "Tôi bị đau buốt răng hàm dưới dữ dội về đêm, đau giật theo nhịp mạch đập lan lên tận thái dương, uống thuốc giảm đau paracetamol chỉ đỡ một lúc rồi lại đau nhức kinh khủng. Bác sĩ tư vấn giúp tôi.",
            "Răng số 7 của tôi bị vỡ một góc lớn, hôm nay đau nhức không ngủ được, chạm răng trên vào là buốt nhói. Có phải tôi bị viêm tủy rồi không?",
            "Cơn đau răng giật giật lan lên tai và đầu, ngậm nước mát thì thấy dịu đi một chút nhưng bỏ ra là đau buốt dữ dội. Đây là triệu chứng bệnh gì?"
        ],
        "i_title": "Sinh lý bệnh Viêm tủy răng và Áp lực Nội tủy",
        "i_content": "Tủy răng nằm trong buồng tủy cứng chắc không có khả năng co giãn. Khi vi khuẩn xâm nhập gây phản ứng viêm, giãn mạch và tăng tiết dịch rỉ viêm sẽ làm tăng áp lực dữ dội bên trong buồng tủy. Áp lực này chèn ép trực tiếp lên các sợi thần kinh cảm giác (sợi A-delta và sợi C), dẫn đến các cơn đau nhức tự nhiên, dữ dội, đau giật theo nhịp mạch đập và đau tăng mạnh khi nằm xuống do áp lực máu dồn lên vùng đầu mặt.",
        "p_title": "Hướng dẫn Chẩn đoán và Điều trị Viêm Tủy Răng - Bộ Y Tế",
        "p_content": "Chẩn đoán xác định viêm tủy không hồi phục: Cơn đau tự nhiên kéo dài, đau lan tỏa, nghiệm pháp gõ dọc răng đau nhẹ đến vừa. Phác đồ điều trị chuẩn: Gây tê tại chỗ vô cảm, mở buồng tủy dẫn lưu giảm áp lực tức thì, tạo hình và làm sạch hệ thống ống tủy bằng dung dịch bơm rửa NaOCl 2.5-5.25%, đặt thuốc sát trùng tủy, sau đó hàn bít kín khít ống tủy theo không gian 3 chiều bằng Gutta-percha và xi măng trám bít chuyên dụng.",
        "claims": [
            {"claim_id": "c1", "text": "Cơn đau nhức dữ dội tự nhiên về đêm, đau giật theo nhịp mạch đập lan lên thái dương là triệu chứng lâm sàng điển hình của viêm tủy răng không hồi phục [P1], sinh ra do áp lực dịch viêm chèn ép sợi thần kinh trong buồng tủy kín [I1].", "citations": ["I1", "P1"]},
            {"claim_id": "c2", "text": "Phương pháp điều trị triệt để duy nhất theo phác đồ Bộ Y Tế là can thiệp nội nha (chữa tủy): gây tê mở tủy dẫn lưu giảm áp, làm sạch ống tủy và trám bít kín 3 chiều bằng Gutta-percha [P1].", "citations": ["P1"]}
        ],
        "treatment_plan": "Mở buồng tủy cấp cứu dẫn lưu giảm đau, điều trị nội nha lấy sạch tủy viêm nhiễm và hàn bít ống tủy bằng Gutta-percha",
        "home_care": "Có thể dùng thuốc giảm đau kháng viêm tạm thời (theo hướng dẫn của dược sĩ/bác sĩ), chườm mát ngoài má, không ngậm nước nóng hay tự ý nhét các loại thuốc dân gian không rõ nguồn gốc vào lỗ sâu."
    },
    {
        "topic": "Viêm Nướu & Viêm Nha Chu Mạn Tính",
        "keywords": ["viêm nướu", "chảy máu chân răng", "cao răng", "tụt nướu", "lung lay răng", "nha chu"],
        "questions": [
            "Mỗi lần đánh răng hoặc cắn quả táo tôi đều thấy bọt kem có lẫn máu, nướu răng vùng cửa hàm dưới hơi sưng đỏ và miệng có mùi hôi. Tôi đang bị bệnh gì?",
            "Chân răng của tôi có nhiều mảng bám màu vàng nâu cứng chắc, nướu bị sưng phồng và hay chảy máu tự nhiên. Bác sĩ cho hỏi điều trị như thế nào?",
            "Tôi năm nay 45 tuổi, gần đây thấy răng cửa bị thưa ra, nướu tụt để lộ chân răng và hơi lung lay nhẹ khi nhai thức ăn dai. Có chữa được không bác sĩ?"
        ],
        "i_title": "Bệnh sinh Viêm nướu và Tổn thương Dây chằng Nha chu",
        "i_content": "Mảng bám vi khuẩn tích tụ lâu ngày trên bề mặt răng bị vôi hóa bởi muối khoáng trong nước bọt tạo thành cao răng (vôi răng). Độc tố vi khuẩn (LPS) kích hoạt phản ứng miễn dịch tại chỗ gây cương tụ mạch máu, dẫn đến tình trạng nướu sưng nề, chuyển màu đỏ sẫm và rất dễ chảy máu khi có tác động cơ học nhẹ. Nếu không điều trị, quá trình viêm phá hủy biểu mô bám dính, tiêu xương ổ răng và làm đứt các bó sợi dây chằng nha chu nâng đỡ răng.",
        "p_title": "Phác đồ Điều trị Viêm Nướu và Viêm Nha Chu - Bộ Y Tế",
        "p_content": "Chỉ định điều trị viêm nướu và nha chu mạn: Lấy sạch toàn bộ cao răng trên nướu và dưới nướu bằng máy siêu âm, xử lý làm nhẵn bề mặt chân răng (Root planing) để loại bỏ độc tố vi khuẩn. Bơm rửa túi lợi bằng dung dịch sát khuẩn Chlorhexidine 0.12% hoặc Betadine súc họng. Trường hợp túi nha chu sâu trên 5mm có tiêu xương cần chỉ định phẫu thuật nạt túi nha chu hoặc ghép xương nhân tạo.",
        "claims": [
            {"claim_id": "c1", "text": "Tình trạng nướu sưng đỏ, chảy máu khi chải răng và có mùi hôi là dấu hiệu của viêm nướu mạn tính tiến triển sang viêm nha chu [P1], bắt nguồn từ mảng bám vôi răng tích tụ vi khuẩn kích thích cương tụ mạch máu [I1].", "citations": ["I1", "P1"]},
            {"claim_id": "c2", "text": "Phác đồ điều trị nền tảng bắt buộc là lấy cao răng trên và dưới nướu bằng sóng siêu âm, kết hợp xử lý làm nhẵn chân răng và bơm rửa sát khuẩn bằng Chlorhexidine [P1] nhằm ngăn chặn tiêu xương ổ răng [I1].", "citations": ["I1", "P1"]}
        ],
        "treatment_plan": "Lấy cao răng toàn phần bằng máy siêu âm, làm sạch mảng bám dưới nướu, bơm rửa sát khuẩn túi lợi bằng Chlorhexidine",
        "home_care": "Chải răng theo phương pháp Bass cải tiến (nghiêng 45 độ về phía viền nướu), sử dụng chỉ nha khoa hoặc máy tăm nước vệ sinh kẽ răng hàng ngày, súc miệng nước muối sinh lý hoặc dung dịch sát khuẩn chuyên dụng."
    },
    {
        "topic": "Khớp Thái Dương Hàm (TMJ) & Rối loạn Cắn khớp",
        "keywords": ["khớp thái dương hàm", "lục cục khi há miệng", "mỏi hàm", "nghiến răng", "cắn khớp", "há miệng hạn chế"],
        "questions": [
            "Mỗi khi há to miệng để ngáp hoặc nhai thức ăn cứng, vùng trước tai tôi phát ra tiếng kêu lục cục rất rõ, buổi sáng ngủ dậy thấy cơ hai bên quai hàm bị mỏi căng tức. Bác sĩ cho tôi hỏi nguyên nhân và cách chữa?",
            "Tôi bị đau ê ẩm vùng khớp hàm bên phải, há miệng chỉ được khoảng 2 ngón tay là bị kẹt và đau nhói. Tình trạng này có nguy hiểm không?",
            "Chồng tôi bảo ban đêm tôi nghiến răng ken két rất to, sáng dậy hàm dưới mỏi nhừ và răng bị mòn vẹt. Tôi nên đi khám và điều trị ở đâu?"
        ],
        "i_title": "Giải phẫu Cơ sinh học Khớp Thái Dương Hàm và Đĩa khớp",
        "i_content": "Khớp thái dương hàm (TMJ) là khớp lưỡng lồi cầu có đĩa khớp sụn sợi chêm giữa lồi cầu xương hàm dưới và hõm khớp xương thái dương. Tiếng kêu lục cục (clicking) xảy ra khi đĩa khớp bị trật ra trước ở tư thế nghỉ và tự trượt trở lại vị trí bình thường khi lồi cầu dịch chuyển ra trước trong động tác há miệng (trật đĩa khớp có hồi phục). Thói quen cận chức năng như nghiến răng (Bruxism), siết chặt răng hoặc sai lệch cắn khớp là yếu tố nguyên nhân hàng đầu gây quá tải cơ cắn và thoái hóa cấu trúc khớp.",
        "p_title": "Hướng dẫn Chẩn đoán và Điều trị Loạn năng Khớp Thái Dương Hàm - Bộ Y Tế",
        "p_content": "Chẩn đoán Hội chứng loạn năng hệ thống nhai (TMD): Đau vùng cơ nhai, tiếng kêu khớp bệnh lý, hạn chế biên độ há ngậm miệng. Phác đồ điều trị bảo tồn đa mô thức: 1. Liệu pháp hành vi (ăn thức ăn mềm, tránh há miệng quá mức, chườm ấm vùng cơ nhai). 2. Điều trị nội khoa giảm đau cơ (kháng viêm NSAIDs, giãn cơ). 3. Chế tạo máng nhai thư giãn giải áp khớp (Occlusal Splint) đeo ban đêm để bảo vệ men răng và tái định vị lồi cầu.",
        "claims": [
            {"claim_id": "c1", "text": "Tiếng kêu lục cục vùng trước tai kèm mỏi cơ hàm khi ngủ dậy là biểu hiện của rối loạn chức năng khớp thái dương hàm (TMJ) [P1], thường do trật đĩa khớp có hồi phục kết hợp thói quen nghiến răng hoặc quá tải cơ cắn [I1].", "citations": ["I1", "P1"]},
            {"claim_id": "c2", "text": "Theo phác đồ điều trị của Bộ Y Tế, can thiệp ưu tiên hàng đầu là các biện pháp bảo tồn: chế tạo máng nhai giải áp khớp đeo ban đêm [P1], vật lý trị liệu chườm ấm và dùng thuốc giãn cơ giảm đau nếu cần [P1].", "citations": ["P1"]}
        ],
        "treatment_plan": "Khám chuyên khoa Cắn khớp - TMJ, chụp phim ConeBeam CT hoặc MRI khớp nếu cần, làm máng nhai chống nghiến thư giãn cơ",
        "home_care": "Ăn thức ăn mềm, cắt nhỏ thức ăn, tránh nhai kẹo cao su hoặc cắn đồ cứng, không há miệng quá rộng khi ngáp, chườm khăn ấm hai bên góc hàm 15 phút trước khi đi ngủ."
    },
    {
        "topic": "Mòn Ngót Cổ Răng (Abfraction) & Ê Buốt Men Ngà",
        "keywords": ["mòn cổ răng", "ngót cổ răng", "rãnh chữ V", "buốt khi uống nước lạnh", "chải răng ngang"],
        "questions": [
            "Ở sát viền nướu các răng hàm bên ngoài của tôi bị khuyết lõm vào một rãnh hình chữ V, uống nước đá hay hít gió lạnh vào là buốt thấu óc. Đây là bệnh gì và có trám lại được không?",
            "Tôi có thói quen đánh răng kéo ngang thật mạnh, giờ thấy chân răng bị mòn vẹt lộ ngà màu vàng, chạm đầu ngón tay vào thấy buốt nhói. Bác sĩ cho lời khuyên.",
            "Răng số 14 và 15 của tôi bị tụt lợi kèm mòn rãnh sâu ở cổ răng, thức ăn hay giắt vào đó gây ê buốt. Có cách nào chấm dứt tình trạng này không?"
        ],
        "i_title": "Cơ chế Tổn thương Mòn ngót Cổ răng do Lực uốn và Cọ xát Cơ học",
        "i_content": "Tổn thương khuyết hình chêm (Wedge-shaped defect / Abfraction) ở vùng cổ răng là kết quả tổng hợp của: 1. Lực uốn vi thể phát sinh trong quá trình ăn nhai hoặc nghiến răng tập trung ứng suất căng tối đa tại vùng ranh giới men - xê măng (CEJ), làm phá vỡ các trụ men răng. 2. Tác động mài mòn cơ học (Abrasion) từ việc dùng bàn chải lông cứng chải răng theo chiều ngang với lực quá mạnh. Khi lớp men mỏng ở cổ răng bị bong tróc, các ống ngà răng bị hở ra thông trực tiếp với dịch ngà, dẫn truyền kích thích nhiệt và thẩm thấu đến tủy gây triệu chứng ê buốt nhạy cảm.",
        "p_title": "Quy trình Trám Điều trị Mòn Cổ Răng Khuyết Cổ Răng - Bộ Y Tế",
        "p_content": "Chỉ định: Các rãnh khuyết cổ răng gây ê buốt hoặc có nguy cơ phạm tủy. Quy trình kỹ thuật: Vệ sinh làm sạch bề mặt khuyết, không cần khoan thêm mô răng nếu đáy xoang đã sạch. Xoi mòn chọn lọc men răng (Selective etching), bôi hệ thống keo dán dính tự xoi mòn (Self-etch adhesive) có tính đàn hồi cao để hấp thu lực uốn thân răng. Trám bít phục hồi đường viền cổ răng bằng hạt độn Composite dòng chảy hoặc Composite vi thể, đánh bóng chuyển tiếp êm dịu với bờ nướu viền.",
        "claims": [
            {"claim_id": "c1", "text": "Rãnh khuyết hình chữ V ở sát viền nướu gây buốt khi gặp lạnh là tổn thương mòn ngót cổ răng (khuyết hình chêm) [P1], do sự kết hợp giữa ứng suất lực nhai uốn cong thân răng và cọ xát cơ học từ thói quen chải răng ngang [I1].", "citations": ["I1", "P1"]},
            {"claim_id": "c2", "text": "Giải pháp phục hồi tối ưu theo phác đồ kỹ thuật Bộ Y Tế là trám bít che kín các ống ngà lộ bằng vật liệu Composite thẩm mỹ có độ đàn hồi cao [P1], giúp dứt điểm triệu chứng ê buốt và bảo vệ tủy răng [I1].", "citations": ["I1", "P1"]}
        ],
        "treatment_plan": "Trám phục hồi thẩm mỹ cổ răng bằng Composite chuyên dụng có tính đàn hồi, bôi vecni Fluoride chống ê buốt ngà răng",
        "home_care": "Thay đổi ngay thói quen chải răng: dùng bàn chải lông siêu mềm, chải xoay tròn hoặc chải dọc từ nướu xuống răng, sử dụng kem đánh răng chuyên biệt giảm ê buốt có chứa Kali Nitrat hoặc Arginine."
    },
    {
        "topic": "Răng Khôn Mọc Lệch & Viêm Lợi Trùm Răng Khôn",
        "keywords": ["răng khôn", "răng số 8", "lợi trùm", "há miệng đau", "nhổ răng khôn"],
        "questions": [
            "Răng khôn cùng góc hàm dưới bên trái của tôi đang nhú lên, phần nướu trùm lên bị sưng tấy, ấn vào thấy chảy dịch mủ hôi, há miệng nuốt nước bọt rất đau. Tôi có nên nhổ không?",
            "Vùng răng trong cùng của tôi bị sưng phồng đau nhức lan lên tai, góc hàm hơi sưng và sốt nhẹ 37.8 độ. Tôi phải làm gì trước khi đi khám?",
            "Chụp phim thấy răng khôn số 8 hàm dưới mọc đâm ngang 90 độ vào răng số 7 bên cạnh. Hiện tại chưa đau nhiều nhưng có cần nhổ bỏ không?"
        ],
        "i_title": "Bệnh sinh Viêm quanh Thân răng Khôn (Pericoronitis) và Biến chứng Mọc kẹt",
        "i_content": "Răng số 8 là răng vĩnh viễn mọc muộn nhất trên cung hàm (thường từ 18-25 tuổi), thời điểm xương hàm đã ngừng phát triển kích thước dẫn đến tình trạng thiếu chỗ mọc. Răng khôn mọc lệch hoặc mọc ngầm tạo khe hở giữa thân răng và vạt lợi trùm, trở thành túi chứa thức ăn và môi trường lý tưởng cho vi khuẩn yếm khí phát triển gây viêm quanh thân răng cấp tính. Biến chứng nguy hiểm gồm: tiêu xương mặt xa và sâu hỏng răng cối số 7 bên cạnh, viêm mô tế bào khoang cơ cắn và áp-xe góc hàm.",
        "p_title": "Quy trình Phẫu thuật Nhổ Răng Khôn Mọc Lệch Mọc Kẹt - Bộ Y Tế",
        "p_content": "Chỉ định nhổ răng khôn mọc lệch: Răng gây biến chứng viêm lợi trùm tái phát, răng đâm ngang làm tổn thương răng số 7, hoặc có nang thân răng. Quy trình: Khi đang có đợt viêm cấp, ưu tiên bơm rửa làm sạch túi lợi trùm bằng dung dịch sát trùng, dùng kháng sinh và kháng viêm toàn thân cho ổn định. Sau khi hết đợt cấp, tiến hành phẫu thuật mở vạt, cắt xương, chia cắt thân chân răng bằng tay khoan phẫu thuật để lấy răng ra nhẹ nhàng, bơm rửa và khâu đóng vết mổ.",
        "claims": [
            {"claim_id": "c1", "text": "Nướu góc hàm sưng mủ, đau khi nuốt và há miệng hạn chế là triệu chứng viêm lợi trùm răng khôn cấp tính [P1], do túi lợi trùm tạo ổ ứ đọng thức ăn và vi khuẩn yếm khí gây nhiễm trùng quanh thân răng số 8 [I1].", "citations": ["I1", "P1"]},
            {"claim_id": "c2", "text": "Khi đang viêm cấp tính, hướng xử trí trước mắt là bơm rửa sát khuẩn ổ viêm và dùng kháng sinh [P1]; sau khi hết viêm, cần phẫu thuật nhổ bỏ răng khôn mọc lệch để tránh biến chứng hỏng răng số 7 bên cạnh [P1, I1].", "citations": ["I1", "P1"]}
        ],
        "treatment_plan": "Bơm rửa làm sạch ổ viêm lợi trùm bằng Betadine nha khoa, kê đơn kháng sinh - giảm đau ổn định, sau đó chụp X-quang Panorama và phẫu thuật nhổ răng khôn",
        "home_care": "Súc miệng nước muối ấm pha loãng hoặc dung dịch Chlorhexidine sau mỗi bữa ăn, dùng tăm bông tẩm nước muối vệ sinh nhẹ nhàng khe nướu, không dùng vật nhọn chọc vào ổ mủ."
    },
    {
        "topic": "Viêm Quanh Cuống Răng Mạn Tính & Áp-xe Quanh Chóp",
        "keywords": ["viêm quanh cuống", "áp xe chóp răng", "cục mụn ở nướu", "chảy mủ chân răng", "gõ răng đau"],
        "questions": [
            "Ở nướu phía trên chiếc răng cửa từng bị va đập của tôi nổi lên một nốt mụn mủ nhỏ, thỉnh thoảng ấn vào thấy phì dịch mủ màu vàng nhạt nhưng không đau dữ dội. Bác sĩ cho hỏi đây là bệnh gì?",
            "Răng hàm của tôi nhai thức ăn chạm nhẹ vào là thấy thốn buốt tận chân răng, cảm giác răng bị trồi cao hơn các răng khác. Tình trạng này điều trị thế nào?",
            "Chụp phim X-quang răng bác sĩ bảo ở chóp chân răng có vùng thấu quang đen hình tròn lớn (u hạt/nang chân răng). Có phải nhổ răng đi không?"
        ],
        "i_title": "Bệnh sinh Tổn thương Viêm Quanh Cuống và Phản ứng Tiêu xương Vùng Chóp",
        "i_content": "Viêm quanh cuống răng (Periapical periodontitis) là hậu quả trực tiếp của tình trạng tủy hoại tử không được điều trị hoặc điều trị nội nha chưa triệt để. Độc tố vi khuẩn và các chất thoái hóa protein từ ống tủy thoát qua lỗ cuống răng kích hoạt phản ứng viêm hạt ở vùng dây chằng quanh cuống, huy động hủy cốt bào làm tiêu hủy xương bè quanh chóp tạo thành u hạt (Granuloma) hoặc nang quanh chóp (Radicular cyst), hình thành đường rò thoát mủ qua bản xương ra niêm mạc nướu.",
        "p_title": "Hướng dẫn Điều trị Viêm Quanh Cuống Răng - Bộ Y Tế",
        "p_content": "Chỉ định điều trị nội nha bảo tồn răng: Mở tủy, loại bỏ hoàn toàn vật liệu trám tủy cũ hoặc tủy hoại tử, đo chiều dài làm việc chính xác bằng máy định vị chóp. Bơm rửa tạo hình hệ thống ống tủy bằng dung dịch NaOCl 3%, đặt thuốc sát trùng buồng tủy bằng Calcium Hydroxide từ 1-2 tuần để trung hòa độc tố và kích thích tái tạo xương chóp. Sau khi hết tiết dịch và đường rò lành lặn, tiến hành hàn bít kín khít ống tủy. Trường hợp nang chóp lớn không đáp ứng nội nha mới chỉ định phẫu thuật cắt cuống nạo nang.",
        "claims": [
            {"claim_id": "c1", "text": "Nốt mụn mủ ở nướu chân răng kèm cảm giác răng trồi cộm khi nhai là dấu hiệu của viêm quanh cuống mạn tính có đường rò mủ [P1], do vi khuẩn hoại tử từ tủy răng lan qua lỗ chóp chân răng kích hoạt tiêu xương bè [I1].", "citations": ["I1", "P1"]},
            {"claim_id": "c2", "text": "Phác đồ chuẩn của Bộ Y Tế ưu tiên điều trị nội nha bảo tồn: làm sạch ống tủy, đặt thuốc sát khuẩn Calcium Hydroxide kích thích lành xương chóp [P1], chỉ phẫu thuật nạo nang cắt chóp khi điều trị nội nha không đáp ứng [P1].", "citations": ["P1"]}
        ],
        "treatment_plan": "Điều trị tủy lại bảo tồn, đặt thuốc sát trùng cuống răng Calcium Hydroxide, chụp X-quang theo dõi lành xương và hàn bít kín ống tủy",
        "home_care": "Tránh nhai thức ăn cứng bên hàm có răng tổn thương, giữ vệ sinh khoang miệng sạch sẽ, tuyệt đối không tự ý dùng kim chọc vỡ nốt mủ ở nướu."
    },
    {
        "topic": "Nhiễm Màu Men Răng & Phục Hình Răng Sứ Thẩm Mỹ",
        "keywords": ["nhiễm màu răng", "tetracycline", "răng ố vàng", "tẩy trắng răng", "dán sứ veneer"],
        "questions": [
            "Răng của tôi bị ố vàng xỉn màu từ nhỏ do ngày xưa mẹ uống thuốc kháng sinh Tetracycline, đánh răng bằng các loại kem làm trắng không cải thiện được. Tôi nên tẩy trắng hay làm răng sứ?",
            "Men răng tôi không đều màu, có những vệt ngang màu xám tro ở các răng cửa phía trước khiến tôi rất tự ti khi cười. Phương án nào thẩm mỹ và ít mài răng nhất?",
            "Tôi muốn cải thiện nụ cười nhưng sợ bị mài nhỏ răng làm tổn thương tủy. Dán sứ Veneer có phù hợp cho răng nhiễm màu không bác sĩ?"
        ],
        "i_title": "Cơ chế Nhiễm sắc Nội sinh Men Ngà và Sinh cơ học Phục hình Tối thiểu",
        "i_content": "Nhiễm màu nội sinh (Intrinsic staining) do Tetracycline xảy ra khi phân tử kháng sinh liên kết chelat hóa với tinh thể Canxi tạo phức hợp Canxi-Tetracycline Orthophosphate lắng đọng sâu trong cấu trúc ngà răng và men răng trong giai đoạn khoáng hóa răng. Do nằm sâu trong mô cứng ngà răng, phương pháp tẩy trắng răng hóa học chỉ có tác dụng một phần ở mức độ nhẹ độ I-II. Với độ III-IV (nhiễm màu xám sẫm, có dải), giải pháp tối ưu bảo tồn sinh học là mặt dán sứ Veneer với độ dày siêu mỏng 0.3-0.5mm, chỉ mài bề mặt men răng phía ngoài mà không xâm phạm mô ngà sâu hay buồng tủy.",
        "p_title": "Quy trình Kỹ thuật Làm Mặt Dán Sứ (Veneer) và Phục hình Thẩm mỹ - Bộ Y Tế",
        "p_content": "Chỉ định: Răng nhiễm màu nội sinh, răng thiểu sản men, sứt mẻ góc cắn hoặc khe thưa răng cửa. Quy trình: Đánh giá độ nhiễm sắc và đường cười, sửa soạn bảo tồn bề mặt men răng mặt ngoài 0.3 - 0.6mm, lấy dấu cao su độ chính xác cao. Dán dính mặt sứ phục hình bằng hệ thống Resin Cement quang trùng hợp có tính cản quang và màu sắc tùy biến theo màu răng thật. Đảm bảo khít sát đường hoàn tất trên nướu để không gây viêm nướu viền.",
        "claims": [
            {"claim_id": "c1", "text": "Răng nhiễm màu xỉn sẫm do kháng sinh Tetracycline là tổn thương nhiễm sắc nội sinh sâu trong cấu trúc ngà răng [I1], rất khó đáp ứng với các biện pháp tẩy trắng thông thường [P1].", "citations": ["I1", "P1"]},
            {"claim_id": "c2", "text": "Giải pháp thẩm mỹ tiên tiến và bảo tồn mô răng tối đa theo hướng dẫn Bộ Y Tế là dán sứ Veneer mỏng (0.3 - 0.5mm) [P1], chỉ sửa soạn lớp men mặt ngoài mà không làm tổn thương tủy răng [I1, P1].", "citations": ["I1", "P1"]}
        ],
        "treatment_plan": "Khám thẩm mỹ nụ cười, chụp ảnh thiết kế Smile Design, sửa soạn bảo tồn men răng và dán mặt sứ Veneer Emax/Lisi cao cấp",
        "home_care": "Tránh dùng răng cửa cắn đồ quá cứng (càng cua, mở nắp chai), hạn chế thực phẩm có màu đậm (cà phê, trà đặc, nước ngọt có ga), vệ sinh răng miệng bằng chỉ nha khoa đều đặn."
    }

]

# Các mẫu câu mở đầu và xưng hô tự nhiên cho bác sĩ tư vấn
DOCTOR_INTROS = [
    "Chào bạn, qua những triệu chứng bạn vừa chia sẻ cùng các tài liệu và ý kiến hội chẩn chuyên khoa,",
    "Chào bạn, đối với tình trạng răng miệng mà bạn đang gặp phải, tôi xin gửi đến bạn nhận định lâm sàng chi tiết như sau:",
    "Bác sĩ xin chào bạn. Dựa trên mô tả triệu chứng của bạn kết hợp với phác đồ y khoa hiện hành,",
    "Chào bạn, cảm ơn bạn đã gửi câu hỏi tham vấn. Qua phân tích chuyên môn về các biểu hiện bệnh lý của bạn,"
]

def generate_visual_case_augmentation(base_case, sample_idx):
    """Bổ sung dấu hiệu hình ảnh từ Vision Agent 4 vào ca bệnh."""
    teeth_candidates = ["Răng cửa giữa 11, 21", "Răng hàm nhỏ 14, 15", "Răng cối lớn 16, 26, 36, 46", "Răng nanh 13, 23", "Vùng răng cửa hàm dưới 31, 32, 41, 42"]
    tooth = teeth_candidates[sample_idx % len(teeth_candidates)]
    
    vf = {
        "image_type": "clinical_photo",
        "anatomical_regions": ["Cung răng hàm trên", "Viền nướu và kẽ răng"],
        "teeth_involved": [tooth],
        "detected_pathologies": [
            {
                "name": base_case["topic"],
                "location": f"Bề mặt {tooth}",
                "severity": "moderate",
                "visual_evidence": "Đổi màu mô răng, viền nướu cương tụ hoặc khuyết mô men ngà quan sát thấy trên ảnh"
            }
        ],
        "preliminary_impression": f"Quan sát thấy dấu hiệu đặc trưng của {base_case['topic']} tại {tooth}",
        "clinical_urgency": "medium"
    }
    return vf

def build_training_item(topic_case, q_text, sample_idx, has_image=False):
    """Xây dựng 1 cặp input/output chuẩn cho SFT Trainer theo ChatML template."""
    vf = generate_visual_case_augmentation(topic_case, sample_idx) if has_image else None
    
    # 1. Khối ý kiến chuyên khoa hội chẩn
    opinions_block = (
        f"👨‍⚕️ [Ý kiến Chuyên khoa Cắn khớp & TMJ (Agent 2)]:\n"
        f"Đánh giá yếu tố cơ sinh học và tương quan giải phẫu: Cần chú ý đến {topic_case['i_title'].lower()}, "
        f"kiểm soát các lực quá tải cơ học và dự phòng tổn thương mô nâng đỡ quanh răng.\n\n"
        f"👨‍⚕️ [Ý kiến Chuyên khoa Phác đồ Bộ Y Tế (Agent 3)]:\n"
        f"Căn cứ {topic_case['p_title']}: Khuyến nghị áp dụng quy trình chuẩn chuyên khoa, "
        f"ưu tiên can thiệp bảo tồn mô răng thật tối đa và phục hồi chức năng ăn nhai theo đúng phác đồ."
    )

    # 2. Khối chứng cứ tài liệu
    evidence_lines = [
        f"- [I1] (Nội bộ: {topic_case['i_title']}): {topic_case['i_content']}",
        f"- [P1] (Bộ Y Tế: {topic_case['p_title']}): {topic_case['p_content']}"
    ]
    if vf:
        p_desc = vf['detected_pathologies'][0]
        evidence_lines.append(
            f"- [Thị giác & Chẩn đoán hình ảnh]: Nhận định sơ bộ: {vf['preliminary_impression']}. "
            f"Tổn thương phát hiện: {p_desc['name']} tại {p_desc['location']} (dấu hiệu: {p_desc['visual_evidence']}). "
            f"Vùng giải phẫu: {', '.join(vf['anatomical_regions'])}."
        )
    clean_context = "\n".join(evidence_lines)

    # 3. Câu hỏi người bệnh
    user_prompt_content = f"""Ý kiến hội chẩn của các Bác sĩ chuyên khoa:
{opinions_block}

Tài liệu y khoa tham khảo:
{clean_context}

Câu hỏi của người bệnh: "{q_text}"

Bác sĩ hãy tổng hợp các tài liệu và kết quả thị giác trên để viết bài tư vấn nha khoa hoàn chỉnh gửi người bệnh:
1. Nêu 1-3 nhận định cốt lõi (key claims) kèm mã trích dẫn ([I...], [P...]).
2. Trình bày bài tư vấn hoàn chỉnh gửi tới người bệnh (Nhận định tổn thương trên ảnh, Nguyên nhân, Hướng xử trí & Phục hồi, Cảnh báo khi nào cần đi khám).

Chỉ trả về DUY NHẤT một JSON hợp lệ tuân thủ schema:
{{
  "key_claims": [
    {{"claim_id": "c1", "text": "...", "citations": ["I1", "P1"]}}
  ],
  "draft_response": "Nội dung bài tư vấn hoàn chỉnh..."
}}"""

    # 4. Tạo Target Output (JSON hợp lệ 100%)
    intro = random.choice(DOCTOR_INTROS)
    visual_note = f" Qua quan sát hình ảnh lâm sàng bạn gửi kèm, bác sĩ nhận thấy rõ vùng {vf['teeth_involved'][0]} có biểu hiện tổn thương đặc trưng." if vf else ""
    
    draft_body = (
        f"{intro}{visual_note}\n\n"
        f"1. Nhận định tình trạng và Nguyên nhân:\n"
        f"- Hiện tại bạn đang có các triệu chứng của {topic_case['topic']} [P1].\n"
        f"- Về cơ chế bệnh sinh: {topic_case['i_content'][:220]}... [I1]. Đây là nguyên nhân trực tiếp dẫn tới cảm giác khó chịu và tổn thương mà bạn đang gặp phải.\n\n"
        f"2. Phương án Xử trí & Điều trị theo Phác đồ chuyên môn:\n"
        f"- {topic_case['treatment_plan']} theo đúng hướng dẫn kỹ thuật của Bộ Y Tế [P1]. Quy trình này giúp xử lý triệt để ổ nhiễm khuẩn/tổn thương và khôi phục thẩm mỹ cũng như chức năng ăn nhai vững chắc.\n\n"
        f"3. Hướng dẫn Chăm sóc tại nhà & Lời khuyên:\n"
        f"- {topic_case['home_care']}\n"
        f"- Lời khuyên quan trọng: Bạn nên sớm sắp xếp đến phòng khám chuyên khoa Răng Hàm Mặt có uy tín để được bác sĩ thăm khám trực tiếp trên ghế máy và tiến hành điều trị kịp thời, tránh để bệnh tiến triển gây biến chứng phức tạp hơn."
    )

    assistant_json = {
        "key_claims": topic_case["claims"],
        "draft_response": draft_body
    }

    return {
        "messages": [
            {"role": "system", "content": DRAFTING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_content},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False, indent=2)}
        ]
    }


def main():
    print("=" * 70)
    print("🚀 BẮT ĐẦU SINH TẬP DỮ LIỆU SFT CHO AGENT 5 (DRAFTING & SYNTHESIS)")
    print("=" * 70)

    dataset = []
    sample_counter = 0

    # Mỗi chủ đề nhân bản với các biến thể xưng hô, câu hỏi và có/không có ảnh
    for topic_idx, topic in enumerate(CLINICAL_TOPICS):
        base_questions = topic["questions"]
        for q_idx, base_q in enumerate(base_questions):
            # Tạo 30 biến thể câu hỏi bằng cách thay đổi xưng hô, ngữ cảnh thời gian, mức độ
            subjects = ["Tôi", "Em", "Cháu", "Mình", "Mẹ em", "Bố tôi", "Người nhà em", "Bạn em"]
            time_modifiers = [
                "khoảng 3 hôm nay", "mấy ngày gần đây", "từ tuần trước", "mới bị sáng nay",
                "bị tái đi tái lại nhiều lần", "được gần một tháng nay", "vừa mới phát hiện hôm qua"
            ]
            intensity_modifiers = [
                "rất khó chịu", "làm tôi mất ăn mất ngủ", "ảnh hưởng nhiều đến giao tiếp và ăn uống",
                "hơi lo lắng không biết có nguy hiểm không", "muốn biết cách chữa dứt điểm"
            ]

            for s in subjects:
                for tm in time_modifiers[:3]:
                    sample_counter += 1
                    # Biến thể câu hỏi tự nhiên
                    adapted_q = f"{s} {base_q.lower().replace('tôi', s.lower())} Tình trạng này xuất hiện {tm} và {random.choice(intensity_modifiers)}."
                    # 50% số ca có đính kèm ảnh lâm sàng từ Vision Agent 4
                    has_img = (sample_counter % 2 == 0)
                    
                    item = build_training_item(topic, adapted_q, sample_counter, has_image=has_img)
                    dataset.append(item)

    # Đảm bảo tổng số mẫu >= 500
    random.seed(42)
    random.shuffle(dataset)
    target_count = min(len(dataset), 550)
    final_dataset = dataset[:target_count]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in final_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Đã sinh thành công {len(final_dataset)} mẫu dữ liệu SFT chất lượng cao!")
    print(f"📁 Đường dẫn lưu: {OUTPUT_FILE}")
    print(f"📊 Kích thước file: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")

    # Kiểm tra tính hợp lệ của mẫu đầu tiên
    first = final_dataset[0]
    ass_content = json.loads(first["messages"][2]["content"])
    print("\n🔍 Kiểm tra mẫu đầu tiên (Assistant JSON Verification):")
    print(f"- Số key_claims: {len(ass_content.get('key_claims', []))}")
    print(f"- Trích dẫn claim 1: {ass_content['key_claims'][0].get('citations')}")
    print(f"- Độ dài draft_response: {len(ass_content.get('draft_response', ''))} ký tự")
    print(f"- Preview draft_response: {ass_content.get('draft_response')[:150]}...\n")
    print("=" * 70)


if __name__ == "__main__":
    main()
