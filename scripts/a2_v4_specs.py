"""Stage 2 (v4) source-verified specifications and minimal-contrast negatives.

Builds atop `scripts.a2_hard_specs` without modifying Stage 1 frozen specs.
Adds:
  - Cleaned needles aligned with `scripts.a2_v4_cleaning.clean_source_text`
  - Gold wording repairs for pool_index 524, 308, and 404
  - 2 same-chunk `minimal_contrast_negatives` per chunk (80 train + 16 challenge)
    using entity/role/condition/comparison traps in varied WH and yes/no forms.
"""
from __future__ import annotations

from copy import deepcopy

from scripts.a2_hard_specs import evaluation_specs as base_evaluation_specs
from scripts.a2_hard_specs import specs as base_train_specs
from scripts.a2_v4_cleaning import clean_source_text

FACT_TEXT_PATCHES: dict[tuple[int, int], str] = {
    (524, 1): (
        'Chỉ khi các cơ cùng tên ở hai bên có tác dụng đưa tới trước hoặc lui sau cùng co, '
        'hàm dưới mới thực hiện được động tác tới trước hoặc lui sau thẳng trên mặt phẳng dọc giữa.'
    ),
    (308, 1): (
        'Thiết kế thí nghiệm phải dựa trên kết quả thí nghiệm trên động vật '
        'và hiểu biết về diễn biến tự nhiên của bệnh hay vấn đề nghiên cứu.'
    ),
    (308, 2): (
        'Mức độ rủi ro không được vượt quá giới hạn tương xứng '
        'với tầm quan trọng nhân đạo của vấn đề cần giải quyết.'
    ),
    (404, 1): 'Tam giác dịch tễ gồm ba thành phần là vật chủ, tác nhân và môi trường.',
}

MINIMAL_CONTRAST_NEGATIVES: dict[int, list[dict[str, str]]] = {
    # --- Train VI (RHM1: 20 chunks x 2 = 40) ---
    520: [
        {
            'question': 'Trong động tác há ngậm thông thường, đường ngang qua hai lỗ hàm đi qua phần nào của cơ chân bướm ngoài?',
            'review': 'Đoạn nêu trục ngang đi qua gần trung tâm của cơ cắn và cơ chân bướm trong, không nhắc tới cơ chân bướm ngoài.',
        },
        {
            'question': 'Bám tận của cơ chân bướm trong tại hàm dưới có trải trên vùng lớn như cơ cắn không?',
            'review': 'Đoạn chỉ nêu bám tận của cơ cắn trải trên vùng khá lớn, không mô tả độ rộng vùng bám tận của cơ chân bướm trong.',
        },
    ],
    521: [
        {
            'question': 'Đĩa khớp lõm hai mặt có tự hoàn thiện được cơ chế bản lề mà không cần sự dự phần của bộ răng không?',
            'review': 'Đoạn nêu riêng đĩa khớp hoàn thiện cơ chế bản lề giữa hai cấu trúc lồi và bộ răng dự phần quan trọng, không khẳng định độc lập khỏi bộ răng.',
        },
        {
            'question': 'Các thể thụ cảm nằm bên trong lớp mô sợi không mạch máu phủ diện khớp giữ vai trò chính trong định vị hàm dưới phải không?',
            'review': 'Đoạn nhắc thể thụ cảm ở câu đầu và mô sợi phủ diện khớp ở mục 2, không nêu thể thụ cảm nằm trong lớp mô sợi diện khớp.',
        },
    ],
    522: [
        {
            'question': 'Trong các cơ hạ hàm gồm hai cơ chân bướm ngoài, hai cơ nhị thân và các cơ trên móng khác, nhóm cơ nào tạo lực kéo xuống mạnh nhất?',
            'review': 'Đoạn chỉ liệt kê các cơ hạ hàm và mô tả cơ chân bướm ngoài, không so sánh độ mạnh lực kéo giữa chúng.',
        },
        {
            'question': 'Nhánh bó trên bám vào phần trước bao khớp và đĩa khớp (cơ bướm đĩa) có bị cơ chân bướm trong phủ ở mặt ngoài giống thân cơ không?',
            'review': 'Đoạn nêu thân cơ chân bướm ngoài được cơ chân bướm trong phủ ở phần trước phía ngoài, không nói rõ nhánh bướm đĩa có bị phủ hay không.',
        },
    ],
    523: [
        {
            'question': 'Khi hàm dưới mở tối đa, lồi cầu dịch chuyển 13–15 mm diễn ra trước hay đồng thời với quãng 5–8 mm của đĩa khớp?',
            'review': 'Đoạn nêu độ dài quãng dịch chuyển khi há tối đa của lồi cầu và đĩa khớp nhưng không nêu trình tự hay mức độ đồng thời giữa hai quãng này.',
        },
        {
            'question': 'Tác giả Finlay (1965) hay Salaorni và Palla (1994) là người đưa ra thuật ngữ "phức hợp lồi cầu-đĩa khớp" trong tài liệu nguồn?',
            'review': 'Các tác giả được trích cho vận động xoay và quãng dịch chuyển; đoạn chỉ viết chung "người ta coi chúng là một phức hợp".',
        },
    ],
    524: [
        {
            'question': 'Để đưa hàm trượt ra trước bên về bên trái, các cơ nâng bên phải có mức độ tham gia lớn hơn bên trái không?',
            'review': 'Đoạn chỉ nêu thí dụ cụ thể cho bên phải với mức độ nhất định ở một bên, không nêu tỷ lệ so sánh lớn hơn giữa hai bên khi đưa sang trái.',
        },
        {
            'question': 'Trong vận động trượt ra trước bên về bên phải, nhóm cơ hạ và đưa ra sau bên phải có co sớm hơn nhóm cơ đưa ra trước bên trái không?',
            'review': 'Đoạn liệt kê hai nhóm tác động (1) và (2) cùng thực hiện vận động trước bên nhưng không nêu nhóm nào co trước nhóm nào.',
        },
    ],
    525: [
        {
            'question': 'Các phân tử lớn và nhỏ đi vào dịch khớp theo cơ chế vận chuyển chủ động xuất phát từ lớp lót hoạt dịch hay từ lớp mô hoạt dịch dưới?',
            'review': 'Đoạn nêu các phân tử xuất hiện nhờ vận chuyển chủ động và mô tả ba lớp mô hoạt dịch riêng, không chỉ rõ xuất phát từ lớp nào.',
        },
        {
            'question': 'Nguồn cung cấp máu dồi dào của lớp mô hoạt dịch dưới có giúp ngăn lớp lót hoạt dịch bị kẹt giữa các mặt đĩa khớp khi chịu tải không?',
            'review': 'Đoạn nêu lớp lót không bị kẹt ở khớp bình thường và riêng lớp dưới giàu máu, không gắn vai trò chống kẹt cho mạch máu lớp dưới.',
        },
    ],
    526: [
        {
            'question': 'Mức độ căng của lá sau đĩa trên ở tư thế nghỉ có tương đương với khi hàm ở tư thế lồng múi tối đa không?',
            'review': 'Đoạn mô tả tư thế há tối đa và tư thế nghỉ của đĩa khớp, không đề cập tư thế lồng múi tối đa hay độ căng ở tư thế nghỉ của lá sau đĩa trên.',
        },
        {
            'question': 'Khi lồi cầu dịch chuyển ra trước, sự dời về phía sau của đĩa khớp do lá sau đĩa dưới hay lá sau đĩa trên tạo lực kéo chính?',
            'review': 'Đoạn nêu lá dưới không căng giãn như lá trên và giữ ổn định đĩa trên đầu lồi cầu, không so sánh lực kéo giữa hai lá khi lồi cầu ra trước.',
        },
    ],
    555: [
        {
            'question': 'Trong hai yếu tố là độ nghiêng gần của răng và lực của cơ mút, yếu tố nào đóng góp tỷ trọng lớn hơn vào quá trình di gần sinh lý?',
            'review': 'Đoạn liệt kê cả hai yếu tố dự phần và nêu lực di gần chủ yếu do hệ thống sợi trên xương ổ, không so sánh tỷ trọng giữa độ nghiêng và cơ mút.',
        },
        {
            'question': 'Sự bồi đắp xương ở mặt xa và tiêu xương ở mặt gần ổ răng có diễn ra rõ rệt hơn ở răng cối trên so với răng cối dưới không?',
            'review': 'Đoạn nêu chung hiện tượng mô học ở ổ răng và riêng hướng nghiêng răng cối trên/dưới, không so sánh tốc độ thay đổi xương giữa hai hàm.',
        },
    ],
    561: [
        {
            'question': 'Việc răng cửa trên tiếp xúc đường khô - ướt của môi dưới khi phát âm âm /f/ hay /v/ có làm giảm lực tác động lên răng sau lúc nhai nghiền không?',
            'review': 'Đoạn nêu riêng ví dụ phát âm và định hướng lực theo trục răng sau khi nhai nghiền, không liên kết hai chức năng này.',
        },
        {
            'question': 'Các nghiên cứu điện cơ đồ có chứng minh rằng người có khớp cắn lý tưởng không bao giờ bị mỏi cơ khi nhai thức ăn cứng kéo dài không?',
            'review': 'Đoạn nêu dựa trên nghiên cứu lâm sàng và điện cơ đồ rút ra 4 điều tiên quyết cùng cảm giác thoải mái không đau, không bàn mỏi cơ khi nhai cứng kéo dài.',
        },
    ],
    607: [
        {
            'question': 'Trong trường hợp ghi nhận góc Bennett ngay cả khi không có vận động Bennett, thành phần dịch chuyển lập tức (immediate side shift) có xuất hiện không?',
            'review': 'Đoạn nêu hai thành phần vận động sang bên và chú thích góc Bennett thể hiện khi không có vận động Bennett, không xác định trạng thái immediate side shift trong tình huống đó.',
        },
        {
            'question': 'Theo chú thích hình 4-15, khoảng dịch chuyển ra ngoài của lồi cầu làm việc có làm trục quay dời ra sau đúng bằng góc B hay không?',
            'review': 'Hình 4-15 chỉ nêu trục quay ở phía sau các lồi cầu làm việc do dịch chuyển ra ngoài, không quy khoảng dời theo góc B.',
        },
    ],
    608: [
        {
            'question': 'Trong nghiên cứu trên người Việt của Hoàng Tử Hùng và Nguyễn Phúc Diên Thảo, vận động bản lề có ý thức và có hướng dẫn có cùng trung bình 19,98 mm không?',
            'review': 'Đoạn nêu hai cách thực hiện và kết quả khoảng 17–23 mm (trung bình 19,98 ± 2,84 mm), không tách số liệu so sánh từng cách thực hiện.',
        },
        {
            'question': 'Tên gọi "vị trí dây chằng hướng dẫn" của Posselt (1961) có phổ biến hơn "vị trí đĩa khớp hướng dẫn" của Ash và Ramfjord (1995) không?',
            'review': 'Đoạn chỉ nêu thuật ngữ thường dùng nhất là tương quan trung tâm, không so sánh độ phổ biến giữa hai tên gọi phụ còn lại.',
        },
    ],
    610: [
        {
            'question': 'Trong ba vị trí cơ bản gồm vị trí nghỉ, vị trí lồng múi tối đa và vị trí tiếp xúc lui sau, vị trí nào được chọn làm vị trí tham chiếu nhiều nhất?',
            'review': 'Đoạn định nghĩa riêng các vị trí cơ bản và khái niệm vị trí tham chiếu, không chỉ định vị trí cơ bản nào làm tham chiếu phổ biến nhất.',
        },
        {
            'question': 'Vận động tự do có thể khởi phát từ vị trí tiếp xúc lui sau khi các răng không chạm nhau không?',
            'review': 'Đoạn định nghĩa vận động tự do là chưa đạt tới biên và không chạm răng, không nêu khả năng khởi phát từ vị trí tiếp xúc lui sau.',
        },
    ],
    612: [
        {
            'question': 'Khi một người thực hiện động tác há miệng tự do ở tư thế ngồi so với tư thế đứng, đường vạch của điểm răng cửa có khác nhau không?',
            'review': 'Đoạn gộp chung hai tư thế ("Nếu một người ngồi hoặc đứng... sẽ vạch đường 8"), không so sánh sự khác biệt giữa ngồi và đứng.',
        },
        {
            'question': 'Trong hệ thống nhai lành mạnh, tâm quay gần trục D (gần lỗ hàm) khi há tự do có trùng với vị trí cơ (muscular position) khi đụng răng nhẹ trở lại không?',
            'review': 'Đoạn mô tả trục quay khi há tự do và vị trí cơ/tiếp xúc đầu tiên khi đóng nhẹ từ tư thế nghỉ như hai hiện tượng khác nhau, không đồng nhất vị trí hai khái niệm.',
        },
    ],
    627: [
        {
            'question': 'Nếu bệnh nhân chỉ mất răng sau ở một bên cung hàm nhưng vẫn giữ được kích thước dọc, hàm dưới có đóng quá mức làm răng trước dưới tựa mặt trong răng trên không?',
            'review': 'Đoạn nêu điều kiện mất tất cả răng sau gây giảm hoặc mất kích thước dọc, không đề cập trường hợp vẫn còn nguyên kích thước dọc.',
        },
        {
            'question': 'Áp lực đặt lên khớp thái dương hàm khi bệnh nhân đóng hàm ở tư thế đầu cúi về sau có lớn hơn khi nhai bằng các răng sau không?',
            'review': 'Đoạn nêu tư thế đầu đưa tiếp xúc răng ra sau hơn LMTĐ, còn ý áp lực lớn hơn khi nhai bằng răng sau thuộc tình trạng đóng quá mức do mất kích thước dọc.',
        },
    ],
    628: [
        {
            'question': 'Giữa thói quen siết chặt răng và mọc răng khôn thiếu chỗ ở người trẻ, yếu tố nào gây xáo trộn KCTT nghiêm trọng hơn?',
            'review': 'Cả hai đều được liệt kê trong nhóm nguyên nhân có thể gây xáo trộn KCTT, không có so sánh mức độ nặng nhẹ giữa chúng.',
        },
        {
            'question': 'Các tín hiệu áp lực từ răng mòn qua nha chu giúp thiết lập lồng múi ổn định có ngăn được sự thay đổi KCTT do tật đẩy lưỡi khi nuốt không?',
            'review': 'Đoạn nêu riêng hai cơ chế ổn định từ nha chu và danh sách nguyên nhân xáo trộn KCTT (trong đó có đẩy lưỡi), không nêu tác dụng ngăn ngừa qua lại.',
        },
    ],
    643: [
        {
            'question': 'Khi đo vận động tối đa theo chiều ngang và chiều trước sau, mốc đo có sử dụng khoảng cách bờ cắn răng cửa giữa trên và dưới như khi đo độ há không?',
            'review': 'Khoảng cách bờ cắn giữa các răng cửa giữa trên và dưới chỉ được nêu cho đo độ há tối đa (chiều đứng), không nêu mốc đo cho hai chiều còn lại.',
        },
        {
            'question': 'Nếu một người trẻ đạt độ há tối đa trong khoảng 40–50 mm nhưng bị lệch hàm khi vận động thì có vẫn được xếp bình thường theo nghiên cứu Hoàng Tử Hùng và cs (2002) không?',
            'review': 'Đoạn nêu khoảng bình thường 40–50 mm, trung bình thanh niên 49,43 ± 5,39 mm và yêu cầu ghi nhận lệch lạc, không đặt quy tắc kết luận chung khi vừa đạt 40–50 mm vừa bị lệch.',
        },
    ],
    644: [
        {
            'question': 'Khi bệnh nhân không tự nhận biết mình đang bị stress cảm xúc, chấn thương hàm mặt cũ có đóng vai trò rõ ràng hơn trong than phiền chính hiện tại không?',
            'review': 'Đoạn đề cập riêng hai khả năng cần khai thác trong bệnh sử hiện tại và tiền sử RHM, không so sánh mức độ đóng góp giữa chúng.',
        },
        {
            'question': 'Chứng viêm khớp dạng thấp trong tiền sử gia đình có liên hệ trực tiếp với đáp ứng điều trị chỉnh hình trước đây của bệnh nhân không?',
            'review': 'Đoạn tách riêng tiền sử bệnh toàn thân/gia đình và tiền sử can thiệp RHM (chỉnh hình, phẫu thuật, phục hình), không nêu mối liên hệ trực tiếp giữa hai yếu tố.',
        },
    ],
    646: [
        {
            'question': 'Khi đau cấp tính xuất hiện trên nền một đau mạn tính, người bệnh biểu hiện trạng thái trầm cảm rõ hơn hay lo lắng rõ hơn?',
            'review': 'Đoạn nêu riêng đau cấp đi kèm lo lắng, đau mạn gắn trầm cảm/mệt mỏi và khả năng xuất hiện chồng nền, không nêu trạng thái nào trội hơn khi kết hợp.',
        },
        {
            'question': 'Thước VAS dài 100 mm có được dùng chủ yếu cho đánh giá đau một chiều trong khi thang lời khai dùng riêng cho đánh giá nhiều chiều không?',
            'review': 'Đoạn phân biệt nghiên cứu cường độ (một chiều) và nhiều chiều ở câu trên rồi liệt kê VAS và thang lời khai để đánh giá mức độ đau, không gán riêng từng thang cho từng chiều.',
        },
    ],
    653: [
        {
            'question': 'Các test đánh giá đau ở cơ chân bướm ngoài có dễ kích thích phản xạ nôn tương tự thao tác sờ bám tận cơ chân bướm trong không?',
            'review': 'Phản xạ nôn được lưu ý ở phần sờ bám tận cơ chân bướm trong; đối với cơ chân bướm ngoài đoạn chỉ nhắc thực hiện test ở phần dưới, không nêu phản xạ nôn.',
        },
        {
            'question': 'Khi sờ phần trước cơ nhị thân lúc bệnh nhân há miệng có kháng lực, người khám có phải đặt một ngón tay ở trong miệng giống hình 7-21 của cơ chân bướm trong không?',
            'review': 'Kỹ thuật đặt ngón trong/ngoài miệng được nêu riêng cho cơ chân bướm trong ở hình 7-21, không mô tả cách đặt ngón cho cơ nhị thân.',
        },
    ],
    655: [
        {
            'question': 'Khi giữ lực đè ép 2–4 kg trong 1–2 phút ở hai bên đồng thời, bệnh nhân cắn chặt răng sẽ cảm thấy đau tại điểm cò sớm hơn lúc ở tư thế nghỉ phải không?',
            'review': 'Đoạn nêu thông số lực, thời gian, sờ hai bên và yêu cầu khám cả lúc cắn chặt lẫn tư thế nghỉ, không so sánh tốc độ xuất hiện đau giữa hai tư thế.',
        },
        {
            'question': 'Phần nguyên ủy của lớp trong cơ cắn có được sờ qua đường trong miệng tại vùng răng cối lớn 2 và 3 hàm trên giống thân lớp trong không?',
            'review': 'Đoạn nêu sờ nguyên ủy, bám tận và thân của lớp ngoài, còn lớp trong chỉ nêu vùng ngoài và trong miệng ứng với răng cối lớn 2, 3 hàm trên mà không tách riêng nguyên ủy.',
        },
    ],
    # --- Train EN (DPH: 20 chunks x 2 = 40) ---
    151: [
        {
            'question': 'Trong mô hình Medicaid có sự tham gia của công ty bảo hiểm tư nhân, các bác sĩ nha khoa từ chối đăng ký vì mức hoàn trả UCR cho trẻ em thấp hơn chi phí điều trị phải không?',
            'review': 'Đoạn nêu trả UCR cho trẻ để khuyến khích bác sĩ nhận khám và riêng việc nhiều bác sĩ không đăng ký để tránh nhận bệnh nhân nghèo, không nêu mức UCR thấp hơn chi phí.',
        },
        {
            'question': 'Ở những bang chỉ giới hạn Medicaid nha khoa cho trẻ dưới 21 tuổi, chương trình CHIP có bổ sung chi trả nha khoa cho người lớn thu nhập thấp không?',
            'review': 'Đoạn mô tả CHIP là chương trình y tế dành cho trẻ em gia đình vượt chuẩn hỗ trợ bang nhưng không đủ điều kiện bảo hiểm tư, không phủ người lớn.',
        },
    ],
    152: [
        {
            'question': 'Bệnh nhân tại phòng khám áp dụng encounter fee phải tách từng dịch vụ theo ngày có bỏ lịch khám dự phòng nhiều hơn bệnh nhân thuộc kế hoạch capitation không?',
            'review': 'Đoạn nêu hai nhược điểm khác nhau của capitation (ít động lực ưu tiên lịch) và encounter fee (bất tiện, giảm chăm sóc toàn diện), không so sánh tỷ lệ bỏ lịch giữa hai nhóm.',
        },
        {
            'question': 'Khi công ty bảo hiểm cắt giảm mức hoàn trả và thu hẹp quyền kiểm soát điều trị trong managed care, nha sĩ có bắt buộc phải chuyển sang chia nhỏ từng lượt theo encounter fee không?',
            'review': 'Đoạn mô tả lý do phản đối managed care và tách riêng mô tả cơ chế encounter fee, không nêu sự chuyển đổi bắt buộc giữa hai hình thức này.',
        },
    ],
    153: [
        {
            'question': 'Tại phòng khám cash-only hoặc no-insurance, phòng nha có vẫn phải gửi giải thích quyền lợi (EOB) điện tử tới hãng bảo hiểm cùng lúc bệnh nhân thanh toán không?',
            'review': 'Đoạn nêu phòng cash-only thu tiền bệnh nhân để bệnh nhân tự nộp hóa đơn xin hoàn lại; EOB là chứng từ hãng bảo hiểm gửi khi phòng khám trực tiếp nộp claim.',
        },
        {
            'question': 'Nếu cùng lúc xảy ra lỗi nhập nhầm ngày thực hiện và chọn sai mã thủ thuật trong hồ sơ, nha sĩ chủ phòng khám có chia sẻ trách nhiệm pháp lý ngang với chuyên viên vệ sinh răng miệng không?',
            'review': 'Đoạn nêu chuyên viên vệ sinh răng miệng chịu trách nhiệm pháp lý về lựa chọn mã được báo cáo, không nêu quy định chia trách nhiệm với nha sĩ chủ phòng khám.',
        },
    ],
    262: [
        {
            'question': 'Khi lập kế hoạch đánh giá chương trình theo Box 13-1, các chỉ số định tính có phù hợp hơn cho formative evaluation trong giai đoạn lập kế hoạch so với summative evaluation không?',
            'review': 'Đoạn nêu đo lường định tính và định lượng đều quan trọng ngang nhau và định nghĩa riêng formative/summative, không gắn riêng định tính cho formative.',
        },
        {
            'question': 'Theo trình tự câu hỏi của Box 13-1, người lập kế hoạch phải trả lời câu hỏi ai thực hiện đánh giá trước khi xác định những lĩnh vực nha khoa nào được đánh giá đúng không?',
            'review': 'Box 13-1 liệt kê 5 câu hỏi lập kế hoạch và bài viết nêu các ví dụ chỉ số/lĩnh vực đánh giá, không quy định thứ tự thực hiện giữa chúng.',
        },
    ],
    263: [
        {
            'question': 'Khi chấm PLI trên cả bốn mặt xa, ngoài, trong và gần của một răng, điểm của chiếc răng đó được tính theo mặt cao nhất hay lấy trung bình bốn mặt?',
            'review': 'Đoạn nêu PLI đánh giá độ dày mảng bám ở bốn mặt răng vùng nướu với thang 0–3 rồi cộng điểm các răng chia số răng đánh giá, không nêu cách gộp bốn mặt trên một răng.',
        },
        {
            'question': 'Một người bệnh đạt xếp loại Good trên thang OHI-S ở Bảng 13-4 có chắc chắn đạt điểm mảng bám PLI ở mức 1 trên từng mặt răng không?',
            'review': 'Bảng 13-4 chỉ liệt kê khoảng điểm của OHI/OHI-S còn mục PLI chấm độc lập từ 0–3, không có quy tắc quy đổi giữa hai chỉ số.',
        },
    ],
    268: [
        {
            'question': 'Nếu dùng kỹ thuật sàng lọc Type 4 với đè lưỡi nhưng chỉ giới hạn trên sáu răng Ramfjord thì số ca âm tính giả có giảm xuống ngang mức khám Type 3 không?',
            'review': 'Đoạn nêu riêng đặc điểm Type 3/4 và khái niệm 6 răng Ramfjord đại diện toàn hàm, không nêu kết quả kết hợp Type 4 với nhóm răng Ramfjord.',
        },
        {
            'question': 'Mọi chỉ số đo tình trạng không hồi phục (irreversible index) có bắt buộc đồng thời là một chỉ số tích lũy (cumulative index) không?',
            'review': 'Đoạn định nghĩa riêng hai cặp phân loại (simple/cumulative và irreversible/reversible), không khẳng định mọi chỉ số không hồi phục đều là tích lũy.',
        },
    ],
    269: [
        {
            'question': 'Chỉ số chia mỗi răng thành 5 phần trên sáu răng 3, 8, 14, 19, 24, 30 có đòi hỏi tiêu chuẩn Reliability cao hơn chỉ số PCR trong Bảng 13-2 không?',
            'review': 'Đoạn mô tả hai chỉ số và liệt kê bảng chung đặc tính mong muốn của một chỉ số nha khoa (gồm Reliability), không so sánh yêu cầu độ tin cậy giữa hai chỉ số.',
        },
        {
            'question': 'Trong quy trình thực hiện PCR sau khi bôi chất nhuộm màu bờ nướu, điểm mảng bám có được phân trọng số theo ba phần cổ, giữa và rìa cắn giống chỉ số đầu tiên không?',
            'review': 'Chia ba theo chiều đứng cổ-giữa-rìa cắn được mô tả ở chỉ số đầu tiên, còn PCR chỉ đếm và chia số mặt có mảng bám cho tổng số mặt sẵn có.',
        },
    ],
    303: [
        {
            'question': 'Sau khi penicillin được phát hiện vào cuối thập niên 1940, những người nông dân tham gia nghiên cứu Tuskegee có được chuyển sang nhận thuốc này miễn phí không?',
            'review': 'Đoạn chỉ nêu ban đầu chưa có thuốc chữa nên họ nhận chăm sóc y tế/tiền bồi dưỡng và sau đó penicillin được phát hiện cuối thập niên 1940, không nêu việc cấp thuốc cho họ.',
        },
        {
            'question': 'Khi chính quyền địa phương làm trung tâm nghiên cứu tiếp cận nhóm thiểu số mắc đái tháo đường tuýp 2, nghiên cứu đó có phải qua kiểm duyệt tuyên bố sức khỏe của FDA như nghiên cứu khu vực tư nhân không?',
            'review': 'Đoạn nêu riêng FDA rà soát nghiên cứu khu vực tư nhân và vai trò trung tâm nghiên cứu của chính quyền địa phương, không gắn cơ chế duyệt tuyên bố của FDA với chính quyền địa phương.',
        },
    ],
    305: [
        {
            'question': 'Khi một đề cương phải qua thẩm định của cả hội đồng IRB nơi tài trợ lẫn IRB nơi thực hiện nghiên cứu, hội đồng nào giữ quyền quyết định cuối cùng về lựa chọn đối tượng?',
            'review': 'Đoạn nêu đề cương có thể phải qua hơn một IRB và liệt kê các nội dung Báo cáo Belmont (kèm chọn đối tượng), không xác định IRB nào có thẩm quyền sau cùng.',
        },
        {
            'question': 'Báo cáo Belmont có quy định số lượng thành viên tối thiểu của mỗi hội đồng IRB khi thẩm định nguy cơ thiên lệch nghiên cứu không?',
            'review': 'Đoạn nêu ba nguyên tắc cùng nội dung của Báo cáo Belmont và tách biệt bias khỏi misconduct, không đề cập thành phần hay số lượng thành viên IRB.',
        },
    ],
    308: [
        {
            'question': 'Theo bộ nguyên tắc được nêu, nếu nhà nghiên cứu tự đóng vai đối tượng thử nghiệm theo ngoại lệ ở điều 6 thì có được miễn yêu cầu chuẩn bị phương tiện bảo vệ ở điều 8 không?',
            'review': 'Điều 6 chỉ nêu ngoại lệ khi thầy thuốc tự làm đối tượng đối với thí nghiệm có nguy cơ tử vong/tàn tật, còn điều 8 quy định chuẩn bị cơ sở bảo vệ mà không nêu ngoại lệ miễn trừ.',
        },
        {
            'question': 'Theo điều 2, trách nhiệm bảo đảm chất lượng đồng thuận của thành viên chỉ tham gia thu thập số liệu có nhẹ hơn người trực tiếp khởi xướng thí nghiệm không?',
            'review': 'Điều 2 nêu trách nhiệm cá nhân thuộc về mỗi người khởi xướng, chỉ đạo hoặc tham gia thực hiện thí nghiệm và không thể chuyển giao mà không chịu trách nhiệm, không phân cấp nặng nhẹ.',
        },
    ],
    322: [
        {
            'question': 'Khi đã chọn cỡ mẫu ban đầu đủ lớn để bù mất mẫu, kỹ thuật lấy mẫu hệ thống (systematic sampling mỗi đối tượng thứ n) có đạt tính đại diện tốt bằng chọn mẫu ngẫu nhiên không?',
            'review': 'Đoạn định nghĩa systematic sampling và nêu lấy mẫu ngẫu nhiên là cách tốt nhất đảm bảo tính đại diện, không so sánh khi kết hợp bù mất mẫu với systematic sampling.',
        },
        {
            'question': 'Khi khảo sát mô tả bằng email tới giám đốc các chương trình vệ sinh răng miệng, phương pháp phân tầng mẫu (stratifying) có hiệu quả hơn chọn mẫu cụm (cluster) không?',
            'review': 'Đoạn nêu ví dụ khảo sát gửi email giám đốc chương trình, liệt kê các kỹ thuật lấy mẫu và định nghĩa phân tầng mẫu, không so sánh phân tầng với chọn mẫu cụm trong khảo sát này.',
        },
    ],
    325: [
        {
            'question': 'Tỷ lệ tử vong do ung thư miệng cao hơn ở người Mỹ gốc Phi có khiến kết quả thử nghiệm thuốc huyết áp ở nam giới khó khái quát hóa cho phụ nữ hơn không?',
            'review': 'Hai ví dụ minh họa hai vấn đề tách biệt (tính khái quát hóa giữa nam/nữ ở thuốc huyết áp và vấn đề tuyển người yếu thế trong tầm soát ung thư miệng), không có liên hệ nhân quả.',
        },
        {
            'question': 'Bệnh nhân tham gia nghiên cứu ung thư miệng để được điều trị miễn phí có tỷ lệ rút lui khỏi nghiên cứu thấp hơn các nhóm dân số khác phải không?',
            'review': 'Đoạn nêu băn khoăn đạo đức về động cơ tham gia vì không đủ tiền chữa trị và riêng ý mất đối tượng/rút lui ở phần phân tích, không so sánh tỷ lệ rút lui giữa các nhóm.',
        },
    ],
    326: [
        {
            'question': 'Khi đánh giá độ lưu giữ của chất trám bít hố rãnh (sealant) mới, thiết kế quasi-experimental không có đối chứng có thích hợp hơn thiết kế split-mouth trên cùng bệnh nhân không?',
            'review': 'Đoạn lấy sealant mới làm ví dụ cho split-mouth (có đối chứng bên miệng kia) và nêu quasi-experimental dùng khi không thực tế lập đối chứng như hàm ngưng thở khi ngủ, không so sánh dùng quasi-experimental cho sealant.',
        },
        {
            'question': 'Trong thử nghiệm yếu tố 2x2 kết hợp bàn chải và hai loại chỉ nha khoa, mỗi người tham gia có tự đóng vai nhóm đối chứng của chính mình như trong crossover không?',
            'review': 'Đoạn nêu tự làm đối chứng đối với crossover/split-mouth rồi tách sang thiết kế 2x2 factorial gồm bốn nhóm tổ hợp, không nói mỗi người tự đối chứng trong thiết kế 2x2.',
        },
    ],
    328: [
        {
            'question': 'Trong nghiên cứu bệnh-chứng thời kỳ đầu về nhiễm fluor răng (fluorosis), thiết kế mù đôi có loại bỏ được sai lệch phát sinh từ nguồn tài trợ nghiên cứu không?',
            'review': 'Đoạn mô tả nghiên cứu bệnh-chứng fluorosis, lấy thử nghiệm nước súc miệng làm ví dụ làm mù giảm sai lệch nhà nghiên cứu/đối tượng, và tách mục nguồn tài trợ riêng.',
        },
        {
            'question': 'Trong thử nghiệm nước súc miệng tại cơ sở cải huấn, nhóm đối tượng nhận ra mình dùng sản phẩm thương mại thay vì chlorhexidine có báo cáo triệu chứng sai lệch nhiều hơn nhóm dùng giả dược không?',
            'review': 'Đoạn nêu sản phẩm thương mại được đóng chai xanh vị bạc hà giống chlorhexidine để giữ mù và riêng ý người biết dùng placebo sẽ ít tuân thủ quy trình, không so sánh báo cáo giữa sản phẩm thương mại và giả dược.',
        },
    ],
    357: [
        {
            'question': 'Theo quy trình 7 bước kiểm định giả thuyết, nếu tăng ngưỡng α từ 0,01 lên 0,05 thì nguy cơ điều trị sai do lỗi loại I có vượt quá hậu quả của lỗi loại II trong mọi nghiên cứu không?',
            'review': 'Đoạn chỉ nêu lỗi loại I có thể tốn kém hơn lỗi loại II với ví dụ điều trị sai và liệt kê bước chọn α (0,05, 0,01...), không so sánh mọi nghiên cứu khi đổi từ 0,01 lên 0,05.',
        },
        {
            'question': 'Trong Bảng 17-9 khi giả thuyết không H0 thực tế đúng nhưng quy mô mẫu nhỏ, xác suất mắc lỗi loại I (α) có lớn hơn xác suất mắc lỗi loại II (β) không?',
            'review': 'Đoạn chỉ nêu khi cỡ mẫu đủ lớn có thể giảm lỗi loại II đồng thời giữ lỗi loại I nhỏ, không nêu quan hệ độ lớn giữa α và β khi cỡ mẫu nhỏ.',
        },
    ],
    359: [
        {
            'question': 'Khi so sánh tỷ lệ lưu giữ giữa hai nhãn hiệu chất trám, giả thuyết nghiên cứu có hướng (directional) có đòi hỏi ngưỡng α khắt khe hơn giả thuyết không hướng không?',
            'review': 'Đoạn lấy ví dụ hai dạng giả thuyết nghiên cứu và nêu chung ngưỡng p ≤ 0,05 cho nghiên cứu sức khỏe răng miệng, không đặt ngưỡng khác nhau cho giả thuyết có hướng và không hướng.',
        },
        {
            'question': 'Thống kê suy luận dùng cho mẫu khảo sát chuyên viên vệ sinh răng miệng Hoa Kỳ có đạt độ chính xác cao hơn khi kiểm định giả thuyết so với khi ước lượng tham số quần thể không?',
            'review': 'Đoạn nêu thống kê suy luận được dùng cho hai mục đích là kiểm định giả thuyết hoặc ước lượng tham số quần thể từ mẫu, không so sánh độ chính xác giữa hai công dụng.',
        },
    ],
    360: [
        {
            'question': 'Theo bảng đối chiếu thống kê, phép kiểm phi tham số nào (Mann-Whitney rank-sum hay Kruskal-Wallis) nhạy hơn với kích thước hiệu ứng nhỏ khi dữ liệu thuộc thang đo thứ bậc?',
            'review': 'Bảng đối chiếu ghép các phép kiểm phi tham số theo thang đo/số nhóm với phép kiểm tham số tương ứng và đoạn nhấn mạnh tầm quan trọng của effect size nói chung, không so sánh độ nhạy giữa hai phép kiểm.',
        },
        {
            'question': 'Trong nhóm phép kiểm đối với hai phép đo trên cùng một nhóm dữ liệu thứ bậc (Sign và Wilcoxon signed-rank), phương pháp nào đòi hỏi nhiều nghiên cứu lặp lại hơn trước khi thay đổi thực hành?',
            'review': 'Cả hai được ghép chung với paired t-test trong bảng và yêu cầu lặp lại nghiên cứu được nêu chung cho dự án nghiên cứu, không phân biệt giữa Sign và Wilcoxon.',
        },
    ],
    363: [
        {
            'question': 'Khi coi biến rời rạc có nhiều giá trị như chỉ số DMFT như biến liên tục trong phân tích thống kê, giá trị trung bình thập phân (ví dụ 2,5) có gây sai lệch kết luận nhiều hơn trung bình số con 2,3 không?',
            'review': 'Đoạn giải thích cả trung bình số con 2,3 và tính hữu ích của trung bình DMFT 2,5 khi xử lý như biến liên tục, không bàn về mức sai lệch giữa hai ví dụ.',
        },
        {
            'question': 'Trong các biến phân loại không mang giá trị số, biến nhị phân như giới tính có thích hợp để đánh số thứ tự logic hơn biến nhiều mức như tình trạng kinh tế xã hội (SES) không?',
            'review': 'Đoạn nêu chung biến phân loại không có ý nghĩa đại lượng dù đánh số (ví dụ SES 1,8 vô nghĩa) và định nghĩa riêng biến nhị phân, không so sánh việc đánh số giữa hai loại.',
        },
    ],
    404: [
        {
            'question': 'Trong tam giác dịch tễ gồm vật chủ, tác nhân và môi trường, yếu tố môi trường có vai trò quyết định hơn đặc điểm di truyền của vật chủ khi bệnh chuyển từ mức endemic sang epidemic không?',
            'review': 'Đoạn trình bày riêng tam giác dịch tễ và riêng Bảng 18-1 định nghĩa endemic/epidemic/pandemic, không phân tích yếu tố nào quyết định bước chuyển mức độ dịch.',
        },
        {
            'question': 'Trong ví dụ 9 ca sởi tại hạt Tarrant tháng 8/2013 so với mức 0–2 ca thông thường mỗi năm, sự bùng phát này được quy cho tác nhân cơ học hay điều kiện xã hội của vật chủ?',
            'review': 'Ví dụ sởi ở Tarrant minh họa định nghĩa epidemic trong Bảng 18-1, còn phần tác nhân sinh học/cơ học và đặc điểm xã hội của vật chủ chỉ nêu ví dụ sâu răng và chải răng ở đoạn trên.',
        },
    ],
    478: [
        {
            'question': 'Ở người lớn khỏe mạnh mang nấm Candida trong hệ vi sinh miệng bình thường, việc hút thuốc lá có làm giảm tỷ lệ loét áp-tơ tái phát thông qua ức chế nấm Candida phát triển quá mức không?',
            'review': 'Đoạn nêu riêng hút thuốc liên quan tỷ lệ loét áp-tơ thấp hơn ở mục bệnh lý loét áp-tơ và tỷ lệ tới 60% người lớn khỏe mạnh mang Candida ở mục nấm, không nêu mối liên hệ giữa hút thuốc và Candida.',
        },
        {
            'question': 'Trong các yếu tố thuận lợi khiến Candida phát triển quá mức như hàm giả, ống hít steroid hay kháng sinh phổ rộng, yếu tố nào gây các ổ loét đau phủ giả mạc trắng vàng kèm quầng đỏ rõ nhất?',
            'review': 'Đặc điểm ổ loét đau phủ giả mạc trắng vàng kèm quầng đỏ thuộc phần mô tả loét áp-tơ tái phát chứ không nêu mức độ nặng tương ứng từng yếu tố nguy cơ của Candida.',
        },
    ],
    # --- Challenge (OHA-CH01 & DPH-CH23: 8 chunks x 2 = 16) ---
    27: [
        {
            'question': 'Đối với người được chăm sóc không có vấn đề về trí nhớ hay phán đoán, tài liệu có khuyến cáo người chăm sóc bôi sẵn kem đánh răng rồi đưa bàn chải hay không?',
            'review': 'Biện pháp nhắc nhở hoặc bôi sẵn kem rồi đưa bàn chải được gợi ý trong mục hỗ trợ người có vấn đề trí nhớ hoặc phán đoán, không nêu khuyến cáo cho người không có vấn đề này.',
        },
        {
            'question': 'Khi đặt tay lên tay người được chăm sóc để hướng dẫn bàn chải, người chăm sóc cần giữ vào mu bàn tay hay cổ tay để tạo lực nhẹ?',
            'review': 'Đoạn chỉ hướng dẫn đặt tay rất nhẹ lên tay người đó và hướng dẫn bàn chải, không nêu chi tiết vị trí mu bàn tay hay cổ tay.',
        },
    ],
    28: [
        {
            'question': 'Nếu người được chăm sóc không súc miệng được, lau miệng bằng gạc ẩm tẩm nước súc miệng nha sĩ kê có hiệu quả làm sạch hơn gạc tẩm nước thường không?',
            'review': 'Đoạn nêu riêng cho uống ngụm nước hoặc lau bằng ngụm gạc ẩm khi không súc được và riêng việc nha sĩ kê nước súc miệng, không so sánh hai dung dịch tẩm gạc.',
        },
        {
            'question': 'Nước súc miệng chống vi khuẩn gây bệnh nướu do nha sĩ kê đơn được dùng để ngâm hàm giả tháo lắp qua đêm hay để chải trực tiếp?',
            'review': 'Đoạn nêu nha sĩ có thể kê nước súc miệng giúp ngừa sâu răng hoặc chống vi khuẩn bệnh nướu và hướng dẫn riêng việc ngâm hàm giả vào nước thường (kèm hỏi nha sĩ về viên rửa hàm giả).',
        },
    ],
    29: [
        {
            'question': 'Khi người được chăm sóc xuất hiện vấn đề hành vi, cách tiếp cận tell-show-do có hiệu quả ổn định hơn việc dùng đồ vật yêu thích hoặc âm nhạc không?',
            'review': 'Tell-show-do và dùng đồ vật yêu thích/âm nhạc là hai gợi ý độc lập trong quy trình hướng dẫn chăm sóc, đoạn không so sánh hiệu quả giữa chúng.',
        },
        {
            'question': 'Khi áp dụng kỹ thuật hai bàn chải (two-toothbrush technique), người chăm sóc nên đứng phía trước hay phía sau để giữ má và chải thuận lợi nhất?',
            'review': 'Các vị trí ngồi hoặc đứng phía trước, bên cạnh hay phía sau được nêu chung ở phần chọn tư thế quan sát mọi mặt răng nướu, không chỉ định riêng cho kỹ thuật hai bàn chải.',
        },
    ],
    30: [
        {
            'question': 'Bàn chải nhiều mặt (multiple-sided toothbrush) có làm sạch mảng bám ở vùng răng sau tốt hơn bàn chải điện (power toothbrush) không?',
            'review': 'Cả hai loại bàn chải cùng được liệt kê như lựa chọn thay thế khi điều chỉnh bàn chải, không so sánh hiệu quả làm sạch vùng răng sau giữa chúng.',
        },
        {
            'question': 'Khi sử dụng miếng xốp kê miệng (foam mouth rest) giữa các răng sau để chống cắn, dụng cụ này có cần tháo ra rửa hằng tuần như phần bọc tay cầm bàn chải không?',
            'review': 'Yêu cầu tháo rửa ít nhất mỗi tuần một lần và để khô hoàn toàn được nêu trong mục TIP cho phần bọc tay cầm và cán bàn chải làm to, không nêu cho foam mouth rest.',
        },
    ],
    6: [
        {
            'question': 'Theo các định nghĩa trong đoạn, một nghiên cứu đạt giá trị bên ngoài (external validity) cao có tự động bảo đảm đạt giá trị bên trong (internal validity) cao không?',
            'review': 'Đoạn định nghĩa riêng internal validity và external validity trong mục validity, không khẳng định đạt giá trị bên ngoài sẽ bảo đảm giá trị bên trong.',
        },
        {
            'question': 'Độ lệch chuẩn (standard deviation) cho biết mức độ tiến gần về giá trị 1 của mối liên hệ tuyến tính giữa hai biến có mạnh hay không phải không?',
            'review': 'Standard deviation thuộc nhóm đo độ phân tán điểm số trong phân phối, còn đánh giá độ mạnh mối liên hệ tuyến tính gần 1 thuộc định nghĩa correlation.',
        },
    ],
    8: [
        {
            'question': 'Trong kỳ thi board quốc gia gồm khoảng 350 câu hỏi, 150 câu theo ca bệnh có được tính trọng số điểm cao hơn 200 câu theo môn học không?',
            'review': 'Đoạn nêu số lượng tổng và số câu của hai thành phần (200 câu môn học và 150 câu ca bệnh), không nêu trọng số tính điểm giữa hai phần.',
        },
        {
            'question': 'Thí sinh có thể tra cứu nội dung cập nhật của kỳ thi ngay trên trang web của ủy ban phát triển đề thi hay chỉ xem được trên trang của ADA?',
            'review': 'Đoạn nêu tra cứu thông tin cập nhật về nội dung và hình thức kỳ thi tại www.ada.org của American Dental Association, không nêu cổng tra cứu của ủy ban đề thi.',
        },
    ],
    11: [
        {
            'question': 'Trong quy trình chăm sóc vệ sinh răng miệng của chương trình cộng đồng, giai đoạn chẩn đoán (diagnosis stage) có bắt buộc phải đặt tách rời khỏi giai đoạn lập kế hoạch không?',
            'review': 'Đoạn nêu nhiều mô hình y tế công cộng đặt giai đoạn chẩn đoán nằm trong giai đoạn lập kế hoạch chứ không nêu quy định bắt buộc tách rời hay không.',
        },
        {
            'question': 'Việc duy trì tiếp xúc cá nhân với quần thể đích có giúp giảm sự phản đối chương trình fluor hóa nước cộng đồng về khía cạnh quyền tự do cá nhân không?',
            'review': 'Hai ý nằm ở các gạch đầu dòng độc lập (phản đối fluor hóa nước về quyền con người/tự do cá nhân và lợi ích của tiếp xúc cá nhân giúp tăng tham gia chương trình), không liên kết nhân quả.',
        },
    ],
    12: [
        {
            'question': 'Khi lập kế hoạch chương trình sealant cho học sinh từ lớp hai đến lớp sáu, sự ủng hộ của lãnh đạo quần thể đích có vai trò quan trọng hơn sự ủng hộ của phụ huynh không?',
            'review': 'Đoạn nêu riêng việc mời lãnh đạo quần thể đích để có hỗ trợ và nhấn mạnh sự ủng hộ của phụ huynh là tối quan trọng nếu chương trình liên quan phụ huynh, không so sánh hơn kém giữa hai bên.',
        },
        {
            'question': 'Tại khu vực có nhu cầu chăm sóc cao nhưng nhu cầu thực hiện thấp, chỉ riêng hoạt động giáo dục sức khỏe răng miệng (dental health education) có làm giảm cả prevalence lẫn incidence ngay trong cùng thời điểm không?',
            'review': 'Đoạn phân biệt education với promotion và định nghĩa riêng prevalence với incidence ở các mục khác nhau, không nêu education đơn thuần làm giảm đồng thời hai chỉ số tại một thời điểm.',
        },
    ],
}


def _patch_specs(raw_specs: list[dict]) -> list[dict]:
    patched = []
    for sp in raw_specs:
        item = deepcopy(sp)
        pidx = item['pool_index']
        if pidx not in MINIMAL_CONTRAST_NEGATIVES or len(MINIMAL_CONTRAST_NEGATIVES[pidx]) != 2:
            raise ValueError(f'Missing 2 minimal_contrast_negatives for pool_index={pidx}')
        for f_idx, fact in enumerate(item['facts']):
            fact['needle'] = clean_source_text(fact['needle'])
            if (pidx, f_idx) in FACT_TEXT_PATCHES:
                fact['text'] = FACT_TEXT_PATCHES[(pidx, f_idx)]
        item['minimal_contrast_negatives'] = deepcopy(MINIMAL_CONTRAST_NEGATIVES[pidx])
        patched.append(item)
    return patched


def v4_train_specs() -> list[dict]:
    return _patch_specs(base_train_specs())


def v4_challenge_specs() -> list[dict]:
    return _patch_specs(base_evaluation_specs())
