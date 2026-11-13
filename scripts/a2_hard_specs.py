"""Source-read hard questions; pool ordinals are resolved to pinned chunk IDs at build."""


def specs():
    out=[]
    def add(i,negative,reason,*facts):
        out.append(dict(pool_index=i,negative_question=negative,negative_review=reason,
                        facts=[dict(question=q,text=t,needle=n) for q,t,n in facts]))
    add(520,'Đoạn này hướng dẫn điều trị rách cơ cắn bằng phương pháp nào?', 'Chỉ mô tả cấu trúc, hướng sợi và vận động, không có xử trí rách cơ.',
        ('Khi xét cấu tạo cơ cắn, có mấy lớp cần phân biệt?','Cơ cắn có lớp ngoài và lớp trong.','Cơ có hai lớp:'),
        ('Các lớp cơ cắn cùng kết thúc ở vùng nào của hàm dưới?','Cả hai lớp bám tận ở vùng góc hàm, lớp ngoài phủ lớp trong.','Cả hai lớp cơ có bám tận'),
        ('Cơ cắn chủ yếu giúp hàm thực hiện động tác gì?','Tác dụng chính của cơ cắn là đóng hàm.','Ngoài tác dụng chính là đóng hàm'))
    add(521,'Đoạn này nêu xét nghiệm nào để xác định viêm khớp thái dương hàm?', 'Đặc điểm giải phẫu và liên thuộc vận động, không nêu xét nghiệm viêm.',
        ('Lớp phủ diện khớp thái dương hàm thuộc mô sụn hay mô sợi?','Các diện khớp được phủ bởi mô sợi không mạch máu, không phải mô sụn.','Các diện khớp được bao phủ'),
        ('Hai khớp thái dương hàm có tự vận động riêng rẽ được không?','Hai khớp độc lập về giải phẫu nhưng không thể thực hiện vận động độc lập.','Các khớp thuộc một hệ thống'),
        ('Bộ phận nào ngoài các cấu trúc khớp cũng dự phần quan trọng vào hoạt động khớp?','Bộ răng dự phần quan trọng vào hoạt động và tình trạng của khớp.','Bộ răng dự phần quan trọng'))
    add(522,'Tài liệu quy định khi nào cần phẫu thuật cơ chân bướm ngoài?', 'Có nguyên ủy, bám và tác động co cơ; không có chỉ định phẫu thuật.',
        ('Bó nào của cơ chân bướm ngoài nhỏ hơn bó còn lại?','Bó trên nhỏ hơn bó dưới.','bó này nhỏ hơn bó dưới'),
        ('Hai bó cơ chân bướm ngoài bám chung ở đâu và bó trên có nhánh bám thêm vào đâu?','Hai bó bám vào hố cơ chân bướm ở mặt trước cổ lồi cầu; một nhánh bó trên bám phần trước bao khớp và đĩa khớp.','Cá hai bó này:'),
        ('Khi chỉ một bên cơ chân bướm ngoài co, hàm có thể được đưa theo những hướng nào?','Co cơ có tác dụng đưa hàm ra trước, xuống dưới và sang bên khi chỉ một bên co.','Khi co, có tác đụng'))
    add(523,'Đoạn này đưa ra cách phục hồi đĩa khớp bị rách như thế nào?', 'Vận động và số đo sinh lý, không mô tả điều trị rách đĩa.',
        ('Cơ có trực tiếp điều khiển phần lớn vận động đĩa trong khe khớp không?','Các vận động của đĩa trong khe khớp nói chung bị động, không có can thiệp trực tiếp của cơ.','Các vận động của đĩa trong khe khớp'),
        ('Vì sao đĩa có thể tự dịch chuyển để vừa với khe khớp thay đổi?','Các mặt trơn và tính đàn hồi giúp đĩa dịch chuyển bị động khi hình dạng khe khớp thay đổi.','Do các mặt trơn'),
        ('Khi há tối đa, quãng dịch chuyển của lồi cầu và đĩa được nêu khác nhau ra sao?','Lồi cầu vận động 13–15 mm, còn đĩa khớp dịch chuyển về trước 5–8 mm.','Toàn bộ đoạn vận động'))
    add(524,'Tài liệu cho biết tổn thương thần kinh nào gây mất động tác đưa hàm lui sau?', 'Mô tả phối hợp cơ nhưng không chẩn đoán dây thần kinh tổn thương.',
        ('Hai cơ thái dương sau cùng tác động sẽ đưa hàm về hướng nào?','Tác động hiệp đồng của hai cơ thái dương sau làm hàm lui về sau.','Tác động hiệp đồng'),
        ('Muốn đưa hàm tới hoặc lui thẳng trên mặt phẳng dọc giữa cần điều kiện phối hợp gì?','Các cơ cùng tên ở hai bên cùng co để đưa hàm tới trước hoặc lui sau thẳng trên mặt phẳng dọc giữa.','Chỉ khi các cơ cùng tên'),
        ('Ở bên làm việc, nhóm cơ nào góp phần tạo vận động sang bên?','Các cơ nâng và cơ đưa ra sau của bên làm việc góp phần tạo vận động sang bên.','Các cơ nâng và cơ đưa ra sau'))
    add(525,'Đoạn này hướng dẫn dùng thuốc nào để tăng tiết dịch khớp?', 'Mô học và cơ chế bôi trơn, không kê thuốc tăng dịch.',
        ('Lớp mô hoạt dịch nằm dưới lớp lót khác về mạch máu như thế nào?','Lớp mô hoạt dịch dưới có nguồn cung cấp máu dồi dào.','Lớp thứ hai là'),
        ('Lớp ngoài cùng trong mô tả ba lớp được tạo bởi thành phần gì?','Lớp thứ ba là bao khớp, gồm các dải collagen dày, không tế bào.','Lớp thứ ba là'),
        ('Dịch khớp đi từ mao mạch ra khoảng gian bào chủ yếu qua cơ chế nào được nêu?','Dịch khớp có bản chất huyết thanh và thoát ra theo khuếch tán thụ động do chuyển dịch mô hoạt dịch.','Dịch khớp có bản chất'))
    add(526,'Đoạn này đặt ra tiêu chuẩn chẩn đoán sai vị trí đĩa khớp bằng MRI nào?', 'Có vị trí, mô và lực kéo; không có tiêu chuẩn hình ảnh MRI.',
        ('Khi há tối đa, mô sau đĩa kéo đĩa theo hướng nào để tránh sai vị trí?','Mô sau đĩa căng, kéo và xoay đĩa ra sau để tránh sai vị trí ra trước.','Ehi há tối đa'),
        ('Đĩa xoay ra trước nhiều nhất ở tư thế nào được mô tả?','Ở tư thế nghỉ, đĩa khớp nằm ở vị trí xoay ra trước nhất.','Ở tư thế nghỉ, đĩa'),
        ('Lá sau đĩa dưới có khả năng giãn giống lá trên không?','Lá sau đĩa dưới không có khả năng căng giãn như lá sau đĩa trên.','Lá sau đĩa đưới không'))
    add(555,'Đoạn này chỉ định khí cụ chỉnh nha nào để ngăn di gần sinh lý?', 'Giải thích di gần và lực tác động, không chỉ định khí cụ.',
        ('Di gần giúp cung răng duy trì sự tiếp xúc như thế nào?','Di gần duy trì tính liên tục cung răng và bù mòn men ở tiếp xúc mặt bên.','Sự di gần diễn ra'),
        ('Hai phía gần và xa của ổ răng thay đổi xương theo hướng nào?','Xương được bồi đắp ở mặt xa và tiêu ở mặt gần ổ răng.','Về mặt mô học'),
        ('Nguồn lực chính làm răng di gần được quy cho hệ thống nào?','Lực di gần chủ yếu do hệ thống sợi trên xương ổ, nhất là các sợi ngang vách.','Về lực di gần'))
    add(561,'Đoạn này cho biết cần can thiệp điều trị ở ngưỡng sai lệch khớp cắn bao nhiêu?', 'Tiêu chuẩn chức năng lý tưởng, không có ngưỡng chỉ định điều trị.',
        ('Sự bảo vệ trong khớp cắn lý tưởng là một chiều hay qua lại giữa răng trước và sau?','Răng trước và răng sau bảo vệ lẫn nhau.','Một khớp cắn lý tưởng'),
        ('Để nhai nghiền hiệu quả, lực lên răng sau cần được hướng ra sao?','Lực tác động trên răng sau được hướng theo trục răng.','Lực tác động trên các răng sau'),
        ('Người có khớp cắn lý tưởng được kỳ vọng có cảm giác chủ quan thế nào?','Họ cảm thấy ổn định, thoải mái, không đau hoặc khó chịu do hoạt động hệ thống nhai.','Người có “khớp cắn lý tưởng”'))
    add(607,'Đoạn này hướng dẫn cách đo góc Bennett bằng dụng cụ nào?', 'Có mô tả góc và hình, không có quy trình/dụng cụ đo.',
        ('Chuyển động sang bên có thể kết hợp những thành phần thời gian nào?','Vận động sang bên có thể gồm thành phần lập tức và tăng tiến hay tiệm tiến.','Vận động sang bên có thể gồm'),
        ('Nếu không có vận động Bennett thì góc Bennett có nhất thiết biến mất không?','Không; góc Bennett thể hiện ngay cả khi không có vận động Bennett.','Góc Bennett thể hiện'),
        ('Theo chú thích hình 4-14, trục quay của lồi cầu không làm việc nằm ở đâu?','Trục quay của lồi cầu không làm việc ở tại lồi cầu làm việc.','Trục quay của lồi cầu không làm việc'))
    add(608,'Đoạn này nêu dấu hiệu bệnh lý nào là chống chỉ định vận động bản lề?', 'Định nghĩa, số đo và tên vị trí; không có chống chỉ định.',
        ('Biên độ bản lề trên người Việt trong nghiên cứu được nêu nằm trong khoảng nào?','Khoảng vận động được nêu là 17–23 mm.','Vận động bản lễ được xác định'),
        ('Vị trí lồi cầu thực hiện được bản lề còn có những tên nào?','Đó là tương quan trung tâm, vị trí bản lề hoặc vị trí lui sau.','Vị trí mà lôi cầu thực hiện'),
        ('Tên nào được dùng phổ biến nhất cho vị trí lồi cầu này?','Tên thường dùng nhất là tương quan trung tâm.','Thuật ngữ thường dùng nhất'))
    add(610,'Tài liệu nêu bệnh nào khiến bệnh nhân không thực hiện được vận động tự do?', 'Phân loại và khái niệm vị trí/vận động, không quy bệnh gây mất vận động.',
        ('Một vận động chưa tới biên mà hai hàm không chạm răng được xếp loại nào?','Đó là vận động tự do.','Vận động tự do: là'),
        ('Vị trí cơ bản được gọi là cơ bản vì có tính lặp lại trong hoạt động nào?','Đó là những vị trí và vận động thường gặp, lặp lại trong hoạt động chức năng.','VỊ trí cơ bản và vận động cơ bản'),
        ('Vị trí tham chiếu được dùng để làm gì thay vì chỉ liệt kê các tư thế?','Chúng được dùng để so sánh, đối chiếu với những vị trí và vận động khác.','VỊ trí tham chiếu và vận động tham chiếu'))
    add(612,'Đoạn này chỉ ra cách điều trị khi vị trí tiếp xúc đầu tiên lệch khỏi lồng múi tối đa như thế nào?', 'Sinh lý từ tư thế nghỉ, không đưa phác đồ điều trị lệch tiếp xúc.',
        ('Trong vận động há tự do, lồi cầu chuyển động theo hướng nào?','Lồi cầu di chuyển ra trước và xuống dưới.','Trong vận động há miệng này'),
        ('Điểm tiếp xúc đầu tiên sau tư thế nghỉ có thể chỉ ở gần lồng múi tối đa không?','Có; tiếp xúc đầu tiên cũng có thể ở một nơi gần vị trí lồng múi tối đa.','Tiếp xúc đầu tiên này'),
        ('Tư thế nào thường mở đầu và kết thúc các vận động tự do?','Tư thế nghỉ thường là tư thế bắt đầu và kết thúc vận động tự do.','Tư thế nghỉ thường'))
    add(627,'Đoạn này hướng dẫn phục hồi kích thước dọc bằng vật liệu nào?', 'Hậu quả mất nâng đỡ và hướng lồi cầu, không có vật liệu phục hồi.',
        ('Khi mất nâng đỡ kích thước dọc, động tác đóng này được gọi là gì?','Đó là vận động đóng quá mức của hàm dưới khi mất thành phần nâng đỡ kích thước dọc.','Là vận động đóng của hàm dưới'),
        ('Đóng quá mức gây hệ quả gì lên nha chu răng trên và áp lực ở khớp?','Nha chu răng trên bị quá tải và áp lực lên khớp thái dương hàm lớn hơn khi nhai bằng răng sau.','Hậu quả là nha chu'),
        ('Lồi cầu đi về đâu trong động tác đóng khi mất kích thước dọc?','Lồi cầu di chuyển ra sau và lên trên.','Trong vận động đóng của hàm dưới khi mất'))
    add(628,'Đoạn này quy định cần chỉnh hình theo trình tự nào khi KCTT thay đổi?', 'Yếu tố sinh lý và bệnh lý làm thay đổi, không trình tự chỉnh hình.',
        ('Tư thế lồng múi ảnh hưởng đến vị trí của cấu trúc nào trong hõm khớp?','Vị trí lồi cầu trong hõm khớp phụ thuộc nhiều vào tư thế lồng múi.','VỊ trí của lỗi cầu trong hõm khớp'),
        ('Răng lồng múi có bảo đảm cơ và khớp đều hài hòa không?','Không; lồng múi có thể diễn ra khi khớp thái dương hàm và cơ chưa hài hòa.','Sự lỗng múi có thể'),
        ('Răng đã mòn còn có thể cung cấp tín hiệu giúp giữ tư thế lồng múi ổn định không?','Có; tín hiệu áp lực từ răng qua nha chu vẫn đủ thiết lập tư thế lồng múi ổn định.','Ngay cả khi răng bị mòn'))
    add(643,'Đoạn này hướng dẫn bài tập nào để chữa hạn chế há miệng?', 'Có cách đo và khoảng bình thường, không bài tập chữa hạn chế.',
        ('Khám vận động tối đa của hàm cần xem những chiều nào?','Cần đo chiều đứng, chiều ngang và chiều trước sau.','Cần đo khả năng vận động tối đa'),
        ('Ngoài khả năng đi xa tới đâu, cần ghi nhận thêm điều gì trong vận động hàm?','Cần ghi nhận biên độ và sự lệch lạc của hàm khi vận động.','Ghi nhận biên độ'),
        ('Khoảng há tối đa được đo giữa hai mốc răng nào?','Đo khoảng cách giữa bờ cắn răng cửa giữa trên và dưới.','Độ há tối đa của hàm thường'))
    add(644,'Đoạn này nêu tiêu chuẩn xét nghiệm để xác nhận stress cảm xúc là gì?', 'Bệnh sử và thông tin cần hỏi, không xét nghiệm xác nhận stress.',
        ('Khi lấy bệnh sử vì sao cần tìm hiểu các hoàn cảnh gây stress?','Stress cảm xúc đóng vai trò quan trọng trong rối loạn chức năng hệ thống nhai.','Tình trạng stress cảm xúc'),
        ('Có thể mặc định bệnh nhân tự nhận biết mình đang stress không?','Không; một số bệnh nhân không biết mình đang stress.','Một số bệnh nhân không biết'),
        ('Khi hỏi về điều trị răng hàm mặt cũ, cần tìm hiểu mốc thời gian và kết quả nào?','Cần hỏi đã điều trị cách nay bao lâu và đáp ứng với các điều trị đó.','Trong tiền sử về răng hàm mặt'))
    add(646,'Đoạn này hướng dẫn kê thuốc gì dựa vào số điểm VAS?', 'Đánh giá đau, không ánh xạ điểm VAS sang kê thuốc.',
        ('Đau mạn có đồng nghĩa mọi bệnh nhân đều đau do tâm lý không?','Không; có thể có nguyên nhân khác chưa xác định được.','Không nên cho rằng tất cả bệnh nhân đau mạn'),
        ('Vì sao cần hỏi và đo đau nhiều lần thay vì một lần?','Đau thường là cơ sở chẩn đoán và phản ánh đáp ứng điều trị.','Vì đau thường là cơ sở'),
        ('Trên VAS, hai đầu thước biểu thị mức nào?','VAS dài 100 mm, một đầu không đau, đầu kia đau không chịu nổi.','Để đánh giá cường độ đau'))
    add(653,'Đoạn này quy định thuốc gây tê để sờ cơ chân bướm ngoài là gì?', 'Không thể sờ trực tiếp và có test, không thuốc gây tê.',
        ('Có thể dùng sờ nắn trực tiếp để khám cơ chân bướm ngoài không?','Không thể sờ nắn trực tiếp cơ chân bướm ngoài.','Không thể sờ nắn trực tiếp'),
        ('Cơ nhị thân được chia thành những phần nào?','Cơ nhị thân gồm thân trước và thân sau.','Cơ nhị thân gồm'),
        ('Khi sờ phần trước cơ nhị thân, bệnh nhân cần thực hiện động tác gì?','Phần trước được sờ khi bệnh nhân há miệng có kháng lực.','Phần trước của cơ được sờ'))
    add(655,'Đoạn này xác định đau cơ cắn ở mức nào thì phải phẫu thuật?', 'Cách sờ và phát hiện điểm đau, không ngưỡng phẫu thuật.',
        ('Khi sờ các cơ khám được hai bên, cách so sánh nên tiến hành thế nào?','Sờ nắn bên phải và trái cùng lúc để so sánh hai bên.','Phải sờ nắn ở hai bên'),
        ('Mục tiêu của thao tác sờ cơ cắn là phát hiện những gì?','Tìm điểm cò, vị trí đau và thay đổi độ chắc của cơ.','Sờ cơ cắn để xác định'),
        ('Nên khám cơ cắn ở lúc nghỉ hay cả khi siết răng?','Khám cơ khi cắn chặt răng và ở tư thế nghỉ.','Cơ nên được khám'))
    add(151,'Đoạn này quy định thời hạn giải quyết khiếu nại Medicaid trong bao nhiêu ngày?', 'Có khác biệt quyền lợi và tiếp cận giữa bang, không thời hạn khiếu nại.',
        ('Vì sao quyền lợi nha khoa Medicaid không đồng nhất giữa các bang?','Các bang có quyền sử dụng tiền liên bang theo những cách khác nhau nên phạm vi nha khoa khác nhau.','Because of the states’ freedom'),
        ('Việc chỉ bảo hiểm cho trẻ có mặt trái gì với những người khác?','Nhiều người khác bị bỏ lại mà không có bảo hiểm nha khoa.','However, when only children'),
        ('Một lý do được nêu khiến bác sĩ không đăng ký cung cấp dịch vụ Medicaid là gì?','Nhiều người không đăng ký để không phải tiếp nhận bệnh nhân nghèo.','Many dental practitioners choose'))
    add(152,'Đoạn này cho biết mức phí cụ thể của một lần encounter là bao nhiêu đô la?', 'Giải thích cơ chế và tác động, không có số tiền.',
        ('Vì sao một số nhà cung cấp phản đối managed care?','Họ phản đối vì giảm hoàn trả và giảm quyền kiểm soát điều trị có thể khuyến nghị.','Some dental ­providers do not support'),
        ('Phần lớn HMO có bao gồm chăm sóc nha khoa thường quy không?','Phần lớn HMO có bảo hiểm y tế nhưng không bao gồm nha khoa thường quy.','Most HMOs'),
        ('Encounter fee tính theo loại dịch vụ hay theo lần tiếp nhận?','Kế hoạch encounter fee trả cho mỗi lần tiếp nhận chăm sóc, bất kể người cung cấp dịch vụ.','The encounter fee plan basically'))
    add(153,'Đoạn này hướng dẫn hạn cuối phải nộp claim cho công ty bảo hiểm là ngày nào?', 'Có đường đi claim, EOB và trách nhiệm mã, không hạn nộp.',
        ('Người tự thanh toán thường trả tiền vào thời điểm nào của lần hẹn?','Họ thường trả ngay khi kết thúc cuộc hẹn.','A patient who is on a self-pay'),
        ('Khi phòng nha nộp claim, có thể gửi bằng những hình thức nào?','Phòng nha gửi cho công ty bảo hiểm bằng điện tử hoặc bản giấy.','If a patient is covered by insurance'),
        ('Ai chịu trách nhiệm pháp lý về lựa chọn mã thủ thuật được báo cáo?','Chuyên viên vệ sinh răng miệng chịu trách nhiệm lựa chọn mã và chịu trách nhiệm pháp lý cho lựa chọn đó.','Dental hygienists are responsible'))
    add(262,'Đoạn này liệt kê phác đồ điều trị cụ thể khi phát hiện sâu răng trong khảo sát là gì?', 'Nêu trách nhiệm chuyển điều trị chứ không mô tả phác đồ.',
        ('Khảo sát thấy bệnh thì người thực hiện có nghĩa vụ gì tiếp theo?','Người thực hiện có trách nhiệm cung cấp chuyển tuyến để điều trị.','When disease is identified'),
        ('Đánh giá quá trình nội bộ lúc lập kế hoạch được gọi là loại nào?','Đó là formative evaluation, đánh giá nội bộ quá trình chương trình và thường làm khi lập kế hoạch.','Formative evaluation is'),
        ('Loại đánh giá giá trị chương trình sau triển khai có tên gì?','Đó là summative evaluation.','Summative evaluation is'))
    add(263,'Đoạn này quy định bao lâu phải đo lại PLI sau điều trị?', 'Có mục đích, dụng cụ, điểm và cách tính, không lịch đo lại.',
        ('PLI xem xét đặc tính nào của mảng bám và ở những mặt răng nào?','PLI đánh giá độ dày mảng bám vùng nướu tại mặt xa, ngoài, trong và gần.','Purpose: Assesses thickness'),
        ('Dùng dụng cụ gì để nhận diện mức mảng bám khi đo PLI?','Dùng cây thăm dò nha chu trên từng răng được đánh giá.','Procedure: Use a periodontal probe'),
        ('Từ điểm từng răng, PLI chung được tính bằng phép toán nào?','Cộng điểm tất cả răng rồi chia cho số răng được đánh giá.','Add scores for all teeth'))
    add(268,'Đoạn này quy định bệnh nhân nào bắt buộc phải dùng khám Type 3 thay Type 4?', 'Mô tả phương tiện và độ hợp lệ, không tiêu chuẩn bắt buộc chọn loại khám theo bệnh nhân.',
        ('Type 3 thường sử dụng bộ phương tiện nào?','Type 3 dùng gương miệng, cây thăm khám và chiếu sáng đầy đủ.','The most common type used'),
        ('Loại khám nào có nhiều âm tính giả nhất và dùng phương tiện gì?','Type 4 là loại ít hợp lệ nhất, dùng đè lưỡi và ánh sáng sẵn có, tạo nhiều âm tính giả nhất.','The least valid type uses'),
        ('Vì sao đo một số răng Ramfjord có thể đại diện toàn hàm?','Các răng Ramfjord đại diện cho toàn bộ răng nên có thể khái quát ra toàn hàm.','The Ramjford teeth are six'))
    add(269,'Đoạn này nêu giá trị PCR nào là chỉ định dùng kháng sinh?', 'Chỉ ghi nhận, nhuộm, tính chỉ số; không chỉ định kháng sinh.',
        ('PCR ghi nhận hiện tượng gì trên từng mặt răng?','PCR ghi nhận sự hiện diện mảng bám vi khuẩn trên từng mặt răng.','Purpose: Records presence'),
        ('Sau khi nhuộm, cần kiểm tra vùng nào để thực hiện PCR?','Kiểm tra bờ nướu để tìm mảng bám.','Procedure: After application of disclosing'),
        ('Mẫu số khi tính PCR là số răng hay số mặt răng sẵn có?','Chia số mặt có mảng bám cho tổng số mặt răng sẵn có.','Scoring: Count and record'))
    add(303,'Đoạn này quy định nghiên cứu phải có cỡ mẫu tối thiểu bao nhiêu người để được FDA duyệt?', 'Mô tả giám sát, lý do cấp tài trợ và ví dụ dừng nghiên cứu, không cỡ mẫu duyệt.',
        ('Giám sát của chính phủ kiểm soát việc bán những sản phẩm và đưa ra tuyên bố nào?','Giám sát kiểm soát điều trị, thuốc, thiết bị y tế được bán và các tuyên bố sức khỏe về chúng.','This government oversight provides'),
        ('Lý do về lợi ích kinh tế nào khiến chính phủ được mô tả là cơ chế cấp tài trợ lý tưởng về lý thuyết?','Chính phủ không sản xuất hoặc thu lợi từ phát triển sản phẩm mới.','The government often provides funding'),
        ('Ở cấp địa phương, đơn vị nào có thể trở thành trung tâm nghiên cứu?','Các chính quyền địa phương như sở y tế công cộng có thể làm trung tâm nghiên cứu.','Local governments, such as'))
    add(305,'Đoạn này hướng dẫn hình phạt pháp lý cụ thể cho plagiarism là gì?', 'Liệt kê hành vi sai và tính cố ý, không hình phạt.',
        ('Ba nguyên tắc nào được rút ra từ Belmont trong đoạn này?','Tôn trọng con người, làm điều có lợi và công bằng.','Three principles are identified'),
        ('Để được xem là research misconduct, hành vi phải có tính chất gì?','Research misconduct phải là cố ý.','Like malpractice, research misconduct'),
        ('Bất đồng cách hiểu kết quả có tự động thành misconduct không?','Không; thiên lệch, bất cẩn không cố ý hoặc bất đồng về ý nghĩa kết quả không cấu thành misconduct theo đoạn.','Bias, unintentional negligence'))
    add(308,'Đoạn này quy định mẫu đơn chấp thuận phải dùng ngôn ngữ nào?', 'Đề cập thông tin, trách nhiệm và thiết kế thí nghiệm, không quy định ngôn ngữ của mẫu đơn.',
        ('Trách nhiệm bảo đảm chất lượng đồng thuận có thể giao hẳn cho người khác không?','Đó là trách nhiệm cá nhân không thể chuyển giao mà không chịu trách nhiệm.','It is a personal duty'),
        ('Thiết kế thí nghiệm cần dựa trên cơ sở nào để biện minh cho việc thực hiện?','Dựa kết quả thí nghiệm động vật và hiểu biết diễn biến tự nhiên của bệnh hay vấn đề nghiên cứu.','The experiment should be so designed'),
        ('Giới hạn rủi ro được so với yếu tố nào?','Rủi ro không được vượt mức do tầm quan trọng nhân đạo của vấn đề cần giải quyết quyết định.','The degree of risk to be taken'))
    add(322,'Đoạn này đưa ra công thức xác định cỡ mẫu tối thiểu cho lấy mẫu ngẫu nhiên là gì?', 'Có cách chọn và phân nhóm, không công thức cỡ mẫu.',
        ('Theo đoạn, cách lấy mẫu nào tốt nhất để bảo đảm tính đại diện đầy đủ?','Lấy mẫu ngẫu nhiên.','The best way of ensuring'),
        ('Cơ hội được chọn giữa các cá thể trong lấy mẫu ngẫu nhiên ra sao?','Mỗi đối tượng được chọn độc lập và có cơ hội được chọn như nhau.','Every possible subject'),
        ('Phân ngẫu nhiên người tham gia vào nhóm thử và chứng đem lại lợi ích gì?','Việc phân ngẫu nhiên làm tăng tính hợp lệ.','Randomly selecting subjects'))
    add(325,'Đoạn này cho biết phần mềm thống kê nào phải dùng để phân tích dữ liệu?', 'Thời điểm phân tích và cách chọn phương pháp, không tên phần mềm.',
        ('Tính đa dạng nào khiến y tế công cộng thuận lợi cho tuyển người nghiên cứu?','Y tế công cộng thường phục vụ nhiều quần thể đích đa dạng.','This is true because public health'),
        ('Dữ liệu có nhất thiết chỉ được gom để phân tích sau lúc kết thúc nghiên cứu không?','Không; có thể thu khi kết thúc hoặc ở các khoảng định trước trong nghiên cứu dài.','Once the study has been completed'),
        ('Phương pháp phân tích nên được chọn lúc nào so với khởi đầu nghiên cứu?','Phương pháp phân tích được chọn trước khi nghiên cứu bắt đầu.','The analysis method is selected'))
    add(326,'Đoạn này quy định thời gian washout cho thử nghiệm crossover là bao lâu?', 'Có tính phù hợp của thiết kế và ưu điểm, không thời gian washout.',
        ('Khi người tham gia tự làm đối chứng, hai nhóm được lợi gì về sự tương đồng?','Có thể đạt sự tương đồng nhóm gần như hoàn hảo.','An advantage of using participants'),
        ('Vì sao những thử nghiệm này có thể dùng mẫu nhỏ?','Mỗi người đồng thời thuộc nhóm thử và nhóm chứng.','Another advantage is that'),
        ('Thiết kế thiếu kiểm soát vốn có được xếp là tiếp cận nào?','Đó là tiếp cận quasi-experimental.','A quasi-experimental approach is used'))
    add(328,'Đoạn này cho biết nồng độ chlorhexidine được đem thử là bao nhiêu phần trăm?', 'Có che giấu can thiệp, màu/vị/chai, không nồng độ.',
        ('Trong ví dụ hai loại súc miệng, ai giữ hồ sơ phân loại người nhận?','Dược sĩ giữ hồ sơ loại nước súc miệng từng người nhận.','The pharmacist kept a record'),
        ('Hai dung dịch được làm giống nhau ở những đặc điểm nào?','Cùng màu xanh, vị bạc hà và đựng trong chai giống nhau không đánh dấu.','A pharmacist could dispense'),
        ('Người biết mình dùng placebo có thể ảnh hưởng nghiên cứu qua hành vi nào?','Họ có thể ít tuân thủ quy trình nghiên cứu hơn.','The subjects also would be less likely'))
    add(357,'Đoạn này cho biết cần bao nhiêu người tham gia để bảo đảm không mắc lỗi loại I?', 'Có diễn giải p và hậu quả lỗi, không cỡ mẫu bảo đảm không lỗi.',
        ('Giá trị p nhỏ hơn mang ý nghĩa gì về mức ý nghĩa thống kê theo đoạn?','Giá trị p thấp biểu thị mức ý nghĩa thống kê lớn hơn.','A low p value signifies'),
        ('Khi p lớn hơn 0,05, kết quả được mô tả ra sao?','Kết quả được nói là không có ý nghĩa thống kê, ký hiệu NS.','When the p value is larger'),
        ('Về thay đổi thực hành, lỗi loại I và II có hệ quả đối lập nào được nêu?','Lỗi loại I dẫn đến thay đổi không có căn cứ, còn lỗi loại II duy trì hiện trạng.','Type I errors lead'))
    add(359,'Đoạn này chỉ rõ phải chọn t-test hay ANOVA khi p nhỏ hơn 0,05 không?', 'Có vai trò suy luận và ngưỡng p, không quy tắc chọn phép kiểm từ p.',
        ('Kiểm định giả thuyết nhằm quyết định giữa hai hướng nào?','Tiếp tục tin giả thuyết không hoặc bác bỏ để ủng hộ giả thuyết nghiên cứu.','The object of hypothesis testing'),
        ('Thống kê suy luận đi từ dữ liệu ở đâu để khái quát đến đâu?','Từ dữ liệu mẫu để suy luận hoặc khái quát về quần thể.','Remember that inferential statistics'),
        ('Ngưỡng p thường được chấp nhận có ý nghĩa trong nghiên cứu sức khỏe răng miệng là bao nhiêu?','Theo đoạn nguồn, p bằng hoặc nhỏ hơn 0,05.','The p value commonly accepted'))
    add(360,'Đoạn này cho biết số lần lặp tối thiểu bắt buộc để thay đổi thực hành là bao nhiêu?', 'Nêu cần lặp nhiều nghiên cứu nhưng không ấn định số tối thiểu.',
        ('Một nghiên cứu riêng lẻ có thường đủ giải quyết dứt điểm câu hỏi không?','Hiếm khi; cần nhiều loại nghiên cứu với các mẫu quần thể khác nhau theo thời gian.','A single research project rarely'),
        ('Những yếu tố nào được coi trọng hơn một kết quả chỉ có p và quyết định ý nghĩa thống kê?','Việc lặp lại nghiên cứu và tính kích thước hiệu ứng.','Also, replications and effect size'),
        ('Kết quả từ chỉ một mẫu đã đủ để thay đổi thực hành chưa?','Chưa; kết quả dựa trên một mẫu không đủ bằng chứng để thay đổi thực hành.','Results based on one sample'))
    add(363,'Đoạn này hướng dẫn dùng phép kiểm nào để so sánh DMFT giữa ba nhóm?', 'Phân loại biến và ví dụ DMFT, không chọn phép kiểm so sánh nhóm.',
        ('Vì sao số răng DMFT được xếp là biến rời rạc?','Không thể có một phần của một chiếc răng trong chỉ số đếm đó.','The dental index, decayed'),
        ('Khi biến không có biểu diễn số, nó thuộc nhóm nào?','Đó là biến phân loại.','A variable that has no numeric'),
        ('Gán số theo thứ tự cho biến phân loại có làm giá trị số đó mang ý nghĩa đại lượng không?','Không; các số gán cho biến phân loại không có ý nghĩa định lượng.','Although categorical variables'))
    add(404,'Đoạn này chỉ định biện pháp điều trị đặc hiệu nào cho một pandemic?', 'Mô hình căn nguyên và thuật ngữ dịch tễ, không điều trị pandemic.',
        ('Tiếp cận đa yếu tố cho rằng một bệnh có bao nhiêu loại nguyên nhân?','Bệnh hay tình trạng có nhiều nguyên nhân.','Today, epidemiology is based'),
        ('Tam giác dịch tễ gồm ba thành phần nào cùng tương tác?','Vật chủ, tác nhân và môi trường.','These factors—the host'),
        ('Tác nhân trong mô hình được hiểu là nguyên nhân thuộc những nhóm nào?','Tác nhân là nguyên nhân sinh học hoặc cơ học của bệnh hay tình trạng.','Agent is the biologic'))
    add(478,'Đoạn này hướng dẫn loại thuốc và liều điều trị Candida là gì?', 'Có dịch tễ và yếu tố nguy cơ, không thuốc/liều.',
        ('Đã biết chính xác cơ chế bệnh sinh của loét áp-tơ tái phát chưa?','Cơ chế bệnh sinh chính xác vẫn chưa được biết.','Although various host'),
        ('Candida albicans có vai trò gì trong nhóm loài gây thrush được nêu?','Candidiasis là nhiễm nấm Candida, trong đó Candida albicans là loài phổ biến nhất.','Candidiasis, sometimes referred'),
        ('Người lớn khỏe mạnh có thể mang Candida như hệ vi sinh bình thường ở tỷ lệ tối đa nào được nêu?','Đoạn nguồn nêu có thể tới 60% người lớn khỏe mạnh mang Candida trong hệ vi sinh miệng bình thường.','As many as 60 percent'))
    # Keep negatives close to the same entity/aspect. Do not let every clinical
    # negative be a request for drugs or surgery absent from all positives.
    close_negatives={
      151:('Khi khiếu nại quyền lợi Medicaid, hạn giải quyết được ấn định là bao lâu?','Không có thời hạn khiếu nại trong chunk.'),
      152:('Mỗi encounter được hoàn trả cụ thể bao nhiêu đô la?','Không nêu số tiền hoàn trả.'),
      153:('Claim bảo hiểm phải nộp trước thời hạn nào?','Không có hạn nộp claim.'),
      262:('Một người được phát hiện sâu răng qua khảo sát sẽ được điều trị theo phác đồ cụ thể nào?','Chỉ có trách nhiệm chuyển điều trị, không phác đồ điều trị.'),
      263:('Cần đo lại PLI sau điều trị vào thời điểm nào?','Không có lịch đo lại PLI.'),
      268:('Những bệnh nhân nào bắt buộc khám Type 3 thay vì Type 4?','Không có tiêu chuẩn chọn loại khám theo bệnh nhân.'),
      269:('PCR ở mức nào thì cần chỉ định kháng sinh?','Không có ngưỡng chỉ định kháng sinh.'),
      303:('FDA yêu cầu cỡ mẫu tối thiểu bao nhiêu để phê duyệt nghiên cứu?','Không có cỡ mẫu tối thiểu để phê duyệt.'),
      305:('Người đạo văn trong nghiên cứu phải chịu hình phạt pháp lý nào?','Không có hình phạt pháp lý cụ thể.'),
      308:('Điều kiện nào cho phép người tham gia rút khỏi thí nghiệm sau khi đã đồng thuận?','Không có quy định rút lui trong chunk này; không suy từ yêu cầu chấp thuận ban đầu.'),
      322:('Cỡ mẫu ngẫu nhiên tối thiểu được tính theo công thức nào?','Có cách chọn mẫu, không công thức cỡ mẫu.'),
      325:('Dữ liệu nghiên cứu phải được phân tích bằng phần mềm nào?','Không có tên phần mềm.'),
      326:('Giữa hai giai đoạn crossover phải chờ washout bao lâu?','Không có thời gian washout.'),
      328:('Nước súc miệng chlorhexidine trong thử nghiệm có nồng độ bao nhiêu?','Không có nồng độ chlorhexidine.'),
      357:('Cỡ mẫu và nguy cơ lỗi loại I có quan hệ định lượng cụ thể nào?','Không có công thức định lượng quan hệ này.'),
      359:('Ở mức p dưới 0,05, phải chọn t-test hay ANOVA trong trường hợp nào?','Không có quy tắc chọn phép kiểm từ p.'),
      360:('Một kết luận cần được lặp lại ít nhất mấy lần trước khi đổi thực hành?','Có yêu cầu lặp lại nhưng không số lần tối thiểu.'),
      363:('Khi so DMFT giữa ba nhóm, nên dùng phép kiểm nào?','Không có chỉ dẫn phép kiểm so sánh nhóm.'),
      404:('Điều trị đặc hiệu trong đại dịch pandemic được chỉ định như thế nào?','Có thuật ngữ pandemic nhưng không chỉ định điều trị.'),
      478:('Điều trị Candida bằng thuốc nào và liều bao nhiêu?','Không có thuốc/liều điều trị Candida.'),
      561:('Cần can thiệp điều trị ở ngưỡng sai lệch khớp cắn bao nhiêu?','Không nêu ngưỡng chỉ định điều trị.'),
      520:('Lớp nào của cơ cắn được huy động trước khi bắt đầu há miệng?','Có các lớp và tác dụng nhưng không nêu lớp co đầu tiên.'),
      521:('Vì sao lớp phủ diện khớp thái dương hàm không có mạch máu?','Nêu đặc điểm không mạch máu, không giải thích nguyên nhân.'),
      522:('Hai bó cơ chân bướm ngoài có do những nhánh thần kinh khác nhau chi phối không?','Có nguyên ủy/bám/tác động, không mô tả chi phối thần kinh từng bó.'),
      523:('Khoảng dịch chuyển đĩa 5–8 mm được đo bằng phương pháp nào?','Có số đo và trích tác giả, không mô tả phương pháp đo.'),
      524:('Cơ nào bắt đầu co sớm nhất khi đưa hàm sang bên?','Có tổ hợp cơ theo bên, không có thứ tự khởi phát co.'),
      525:('Lượng dịch khớp sẽ giảm bao nhiêu khi nguồn máu của mô hoạt dịch dưới giảm?','Có nguồn máu và khuếch tán, không quan hệ định lượng này.'),
      526:('Mô sau đĩa bắt đầu căng tại góc há bao nhiêu độ?','Có mô tả há tối đa, không góc bắt đầu căng.'),
      555:('Sự di gần sinh lý dừng lại ở độ tuổi nào?','Có tích tuổi và cơ chế duy trì tiếp xúc, không tuổi kết thúc.'),
      607:('Góc Bennett thay đổi thế nào khi tăng lực co cơ chân bướm ngoài?','Không có quan hệ giữa lực cơ và độ lớn góc.'),
      608:('Muốn tái tạo vận động bản lề trên giá khớp cần làm theo những bước nào?','Định nghĩa và tên vị trí, không quy trình lên giá khớp.'),
      610:('Vị trí tham chiếu nào chính xác nhất cho mọi bệnh nhân?','Có khái niệm tham chiếu, không xếp hạng độ chính xác.'),
      612:('Bộ nhớ cơ được kiểm tra bằng nghiệm pháp cụ thể nào?','Có tên và vai trò cân bằng cơ, không quy trình kiểm tra.'),
      627:('Sau mất răng bao lâu thì hiện tượng đóng quá mức bắt đầu?','Có điều kiện và hậu quả mất nâng đỡ, không thời gian khởi phát.'),
      628:('Vì sao thụ cảm nha chu vẫn duy trì được tín hiệu áp lực khi răng mòn?','Nêu tín hiệu vẫn đủ, không giải thích cơ chế duy trì tín hiệu.'),
      643:('So với người cao tuổi, độ há trung bình của người trẻ khác nhau thế nào?','Không có nhóm cao tuổi để so sánh.'),
      644:('Tình trạng hôn nhân ảnh hưởng rối loạn thái dương hàm theo cơ chế nào?','Có yêu cầu ghi nhận hôn nhân, không giải thích cơ chế tác động.'),
      646:('Mức VAS 50 mm được quy đổi sang đau nhẹ, vừa hay dữ dội theo bảng nào?','Nêu thang VAS và thang lời nói riêng, không có bảng quy đổi.'),
      653:('Test cụ thể nào được dùng để xác nhận đau thật ở cơ chân bướm ngoài?','Chỉ nói có thể làm test và xem phần dưới, không nêu test trong chunk.'),
      655:('Các điểm cò cơ cắn được phân loại thành những mức nào?','Có mục tiêu tìm điểm cò, không hệ thống phân mức.'),
    }
    for s in out:
        if s['pool_index'] in close_negatives:
            s['negative_question'],s['negative_review']=close_negatives[s['pool_index']]
    return out


def evaluation_specs():
    def item(i,n,r,*facts):
        neutral_questions={
          27:'Hôi miệng do mảng bám được điều trị bằng loại thuốc nào?',
          28:'Lịch tái khám nha khoa bắt buộc cách nhau chính xác mấy tháng?',
          29:'Loại nhạc nào hiệu quả nhất để giảm chống đối khi chăm sóc răng miệng?',
          30:'Nhãn hiệu bàn chải điện nào tốt nhất cho người được chăm sóc?',
          6:'Hai lần đo để kiểm tra reliability phải cách nhau bao lâu?',
          8:'Hạn chót đăng ký thi hội đồng nha khoa là ngày nào?',
          11:'Ngân sách tối thiểu để triển khai chương trình là bao nhiêu?',
          12:'Liều fluoride dùng trong chương trình dự phòng là bao nhiêu?',
        }
        n=neutral_questions[i]
        return dict(pool_index=i,negative_question=n,negative_review=r,
                    facts=[dict(question=q,text=t,needle=a) for q,t,a in facts])
    return [
      item(27,'Đoạn này chỉ định loại thuốc nào để điều trị hôi miệng do mảng bám?', 'Có tác hại và hỗ trợ chải, không thuốc điều trị.',
        ('Mảng bám tích tụ có thể gây những vấn đề nào?','Có thể gây sâu răng, bệnh nướu và hôi miệng.','Plaque buildup can cause'),
        ('Người có vấn đề trí nhớ hoặc khả năng phán đoán có thể cần sự hỗ trợ gì khi tự chải?','Họ có thể cần lời nhắc.','If the person you care for has problems'),
        ('Khi giúp điều khiển bàn chải bằng tay, nên hỗ trợ như thế nào?','Đặt tay rất nhẹ lên tay người đó và hướng dẫn bàn chải.','Help brush by placing')),
      item(28,'Theo đoạn này, lịch tái khám nha khoa bắt buộc cách nhau chính xác mấy tháng?', 'Chỉ nói khám thường xuyên, không chu kỳ tháng.',
        ('Nếu kem đánh răng gây khó chịu, đoạn nguồn nêu cách thay thế nào?','Chải với nước thay thế.','If toothpaste bothers'),
        ('Trước buổi khám, nên chuẩn bị các thông tin cá nhân nào?','Danh sách thuốc hiện dùng, dị ứng đã biết và thông tin bảo hiểm hoặc thanh toán.','Before each appointment'),
        ('Nước ngâm hàm giả được thay với tần suất nào?','Thay nước trong cốc hằng ngày.','Change the water')),
      item(29,'Đoạn này quy định loại nhạc nào hiệu quả nhất để giảm chống đối khi chăm sóc?', 'Có thể dùng nhạc yêu thích nhưng không xếp hạng loại nhạc.',
        ('Tell-show-do yêu cầu thứ tự hành động nào?','Nói điều sẽ làm, cho xem rồi thực hiện chăm sóc như đã giải thích.','Tell what will happen'),
        ('Muốn duy trì thói quen chăm sóc, nên ổn định những yếu tố nào mỗi ngày?','Dùng cùng kỹ thuật vào cùng thời gian và địa điểm mỗi ngày.','Use the same technique'),
        ('Trong kỹ thuật hai bàn chải, bàn chải thứ hai dùng để làm gì?','Bàn chải thứ hai dùng để chải răng.','Use second toothbrush')),
      item(30,'Đoạn này cho biết nhãn hiệu bàn chải điện nào tốt nhất?', 'Có các loại bàn chải và cách điều chỉnh, không nhãn hiệu/xếp hạng.',
        ('Vì sao khuyên dùng đầu bàn chải nhỏ?','Đầu nhỏ giúp dễ làm sạch vùng răng sau.','Uses a toothbrush with a small head'),
        ('Tay cầm được bọc to cần vệ sinh ít nhất bao lâu một lần?','Tháo và làm sạch phần bọc cùng tay cầm ít nhất một lần mỗi tuần.','Remove and clean the grip'),
        ('Khi đặt foam mouth rest một bên, cần chải ở phía nào?','Chải phía đối diện của miệng.','Brush teeth on opposite')),
      item(6,'Đoạn này quy định khoảng cách thời gian phải đo lại để kiểm tra reliability là bao lâu?', 'Có định nghĩa nhất quán nhưng không ấn định khoảng đo lại.',
        ('Cùng dụng cụ đo đi đo lại trên cùng quần thể mà cho kết quả nhất quán là thuộc tính nào?','Đó là reliability.','Reliability is the consistency'),
        ('Nghiên cứu thử trước nghiên cứu dài hạn được dùng để làm gì?','Pilot study giúp các biến được kiểm soát tốt nhất trong nghiên cứu dài hạn.','A pilot study is'),
        ('Biến nào là thứ nhà nghiên cứu chủ động thao tác?','Independent variable là biến được thao tác dưới kiểm soát nhà nghiên cứu.','Independent variable is')),
      item(8,'Đoạn này cho biết hạn chót đăng ký thi hội đồng nha khoa là ngày nào?', 'Có cấu trúc bài thi theo sách và nơi tra cứu, không hạn đăng ký.',
        ('Theo sách, phần theo môn học có bao nhiêu câu hỏi?','Phần discipline-based có 200 câu.','The discipline-based component includes'),
        ('Phần còn lại tổ chức câu hỏi theo hình thức nào và có bao nhiêu câu?','Có 150 câu theo ca bệnh.','The second component includes'),
        ('Đoạn nguồn hướng người học đến tổ chức nào để xem nội dung và hình thức thi cập nhật?','Trang web American Dental Association.','Students can get up-to-date')),
      item(11,'Đoạn này nêu ngân sách tối thiểu để triển khai một chương trình là bao nhiêu?', 'Có nguyên tắc lập kế hoạch nhưng không mức ngân sách.',
        ('Vì sao mục tiêu chương trình phải đo được?','Để chương trình có thể được đánh giá và đo lường hiệu quả.','Program planning should include developing measurable'),
        ('Cách nào được nêu có lợi cho tăng sự tham gia của quần thể đích?','Tiếp xúc cá nhân với quần thể đích.','To increase participation'),
        ('Khi lập kế hoạch, quần thể đích có được tham gia không?','Cần bao gồm quần thể đích hoặc đại diện của họ.','Dental hygienists should include the target')),
      item(12,'Đoạn này quy định liều fluoride cần dùng trong chương trình dự phòng là bao nhiêu?', 'Có mục tiêu, tổ chức và sealant nhưng không liều fluoride.',
        ('Mục tiêu chung của các chương trình y tế công cộng nha khoa là gì?','Giảm bệnh răng miệng.','The overall goal of dental public health'),
        ('Trước khi bắt đầu chương trình, nên liên hệ ai để có chấp thuận và hỗ trợ?','Người quản trị đứng đầu.','When initiating a dental public health'),
        ('Chương trình sealant được mô tả nên nhắm lớp nào và vì sao?','Trẻ lớp hai đến lớp sáu để răng cối đã mọc đầy đủ cho đặt sealant.','A dental sealant program should')),
    ]
