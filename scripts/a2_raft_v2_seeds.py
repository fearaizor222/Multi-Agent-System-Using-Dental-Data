"""Source-read train-only additions: distinct facts, not paraphrase multiplication."""


def specifications():
    result=[]
    def add(name,parent,lang,q0,q1,both,facts):
        result.append(dict(id=name,parent_id=parent,language=lang,questions=[q0,q1],
            combined_question=both,facts=facts))
    # Each fact supplies Vietnamese wording, parent evidence ID and exact span endpoints.
    add('reversals','a2v2_432016f015af27da','en',
        'Theo tài liệu, positive reversal là thay đổi theo hướng nào?',
        'Theo tài liệu, negative reversal là thay đổi theo hướng nào?',
        'Theo tài liệu, positive reversal và negative reversal khác nhau thế nào về hướng thay đổi?',[
        ('Positive reversal là thay đổi phép đo do sai sót theo hướng hợp lý.','I1','Positive reversal Change','in a logical direction'),
        ('Negative reversal là thay đổi chẩn đoán theo hướng bất hợp lý qua một khoảng thời gian.','I1','Negative reversal Change','over a period of time')])
    add('hypotheses','a2v2_e9f26cc8d4f0135c','en',
        'Trong ví dụ kem đánh răng nhãn hiệu X, giả thuyết nghiên cứu phát biểu gì?',
        'Trong ví dụ kem đánh răng nhãn hiệu X, giả thuyết không phát biểu gì?',
        'Trong ví dụ kem đánh răng nhãn hiệu X, giả thuyết nghiên cứu và giả thuyết không được phát biểu như thế nào?',[
        ('Giả thuyết nghiên cứu cho rằng nhãn hiệu X làm trắng răng đáng kể.','I2','Research (Positive) Hypothesis: Brand X','whitens teeth.'),
        ('Giả thuyết không cho rằng không có khác biệt có ý nghĩa thống kê giữa nhãn hiệu X và giả dược về khả năng làm trắng răng.','I2','Null Hypothesis: There is no statistically significant','the whitening of teeth.')])
    add('dental_board','a2v2_64187f7e35d89708','en',
        'Theo đoạn nguồn, nhiệm vụ nào của Dental Board cần được cân nhắc khi thay đổi quy định?',
        'Theo đoạn nguồn, nhà vận động hành lang giúp soạn yêu cầu thay đổi theo loại ngôn ngữ nào?',
        'Khi thay đổi quy định, cần cân nhắc nhiệm vụ nào của Dental Board và nhà vận động hành lang giúp soạn yêu cầu theo ngôn ngữ nào?',[
        ('Nhiệm vụ của Dental Board cần được cân nhắc là bảo vệ công chúng.','I1','The most important thing to consider','to protect the public.'),
        ('Nhà vận động hành lang giúp soạn yêu cầu theo ngôn ngữ pháp lý thường dùng trong luật hành nghề nha khoa.','I1','Once the professional association determines','act.')])
    add('global_care','a2v2_48d5ee539914ff15','en',
        'Theo tài liệu, chuyên viên vệ sinh răng miệng kết hợp những hoạt động nào để thúc đẩy lối sống lành mạnh?',
        'Theo tài liệu, mô hình mobile dental care vận chuyển cơ sở chăm sóc đến nhóm dân cư nào?',
        'Theo tài liệu, những hoạt động nào thúc đẩy lối sống lành mạnh và mobile dental care vận chuyển cơ sở chăm sóc đến nhóm dân cư nào?',[
        ('Các hoạt động gồm chương trình cai thuốc lá và tư vấn dinh dưỡng trong thực hành.','I1','Dental hygienists have also been effective','into their practices.'),
        ('Mobile dental care vận chuyển cơ sở chăm sóc răng miệng đến nhóm dân cư không di chuyển được.','I2','Autonomous dental hygiene care','to a nonmobile population.')])
    add('payment_systems','a2v2_5c55b00999817727','en',
        'Trong hệ thống thanh toán hai bên được mô tả, ai trả tiền dịch vụ nha khoa?',
        'Trong hệ thống thanh toán ba bên được mô tả, ai trả tiền dịch vụ nha khoa?',
        'Theo tài liệu, người trả tiền dịch vụ khác nhau thế nào giữa hệ thống hai bên và ba bên?',[
        ('Trong hệ thống hai bên, bệnh nhân trả tiền dịch vụ do nhà cung cấp nha khoa thực hiện.','I1','Two-party system:','for which the patient pays'),
        ('Trong hệ thống ba bên, bên bảo trợ của bệnh nhân, như công ty bảo hiểm hoặc người sử dụng lao động, trả tiền dịch vụ.','I1','Three-party system:','company or employer) pays')])
    add('socioeconomic_status','a2v2_77686b36a76db2e2','en',
        'Theo đoạn nguồn, socioeconomic status được hiểu là vị thế như thế nào của một cá nhân?',
        'Theo đoạn nguồn, tần suất sâu răng chưa điều trị thay đổi ra sao khi SES thấp hơn?',
        'Theo đoạn nguồn, SES là vị thế gì và tần suất sâu răng chưa điều trị thay đổi ra sao khi SES thấp hơn?',[
        ('SES là vị thế xã hội và kinh tế tương đối của một cá nhân trong cộng đồng.','I1','Demographic information shows','within a community.'),
        ('Theo đoạn nguồn, SES càng thấp thì sâu răng chưa điều trị càng thường gặp.','I1','Specifically, the lower','untreated dental caries.')])
    add('presentation_visuals','a2v2_dca4f39da9895fc0','en',
        'Theo tài liệu, tại sao nên tránh trình bày nhiều chữ nghiêng hoặc toàn chữ in hoa?',
        'Theo tài liệu, mỗi ý trên hình trình chiếu nên được giới hạn thế nào để giảm thời gian đọc?',
        'Theo tài liệu, vì sao cần tránh nhiều chữ nghiêng hoặc in hoa và nên giới hạn mỗi ý trên hình trình chiếu thế nào?',[
        ('Nên tránh nhiều chữ nghiêng hoặc toàn chữ in hoa vì chúng làm chậm việc đọc.','I1','avoid putting much text','this slows down reading'),
        ('Khi có thể, giới hạn mỗi ý trong một dòng để giảm thời gian đọc.','I1','limit each point','to limit reading time')])
    add('regression_example','a2v2_4ff294b7b5729bde','en',
        'Trong ví dụ nghiên cứu được nêu, thiết kế theo dõi dọc kéo dài bao lâu và có cỡ mẫu bao nhiêu?',
        'Theo đoạn nguồn, phân tích hồi quy thuộc thống kê mô tả hay thống kê suy luận?',
        'Trong ví dụ được nêu, nghiên cứu theo dõi dọc kéo dài bao lâu, có cỡ mẫu bao nhiêu và hồi quy thuộc loại thống kê nào?',[
        ('Nghiên cứu đoàn hệ dịch tễ học theo dõi dọc kéo dài hơn 20 năm với cỡ mẫu 500.000.','I1','Very strong—this study represents','sample size of 500,000).'),
        ('Phân tích hồi quy là một phương pháp thống kê suy luận.','I1','Although it examines the relationship','regression analysis is an inferential statistic.')])
    add('providers_research','a2v2_aed584cc133ad42c','en',
        'Theo đoạn nguồn, denturists chế tạo và cung cấp loại phục hình nào trực tiếp cho bệnh nhân?',
        'Theo đoạn nguồn, NIDCR tập trung nghiên cứu lĩnh vực nào?',
        'Theo đoạn nguồn, denturists cung cấp loại phục hình nào và NIDCR tập trung nghiên cứu lĩnh vực nào?',[
        ('Denturists chế tạo và cung cấp trực tiếp cho bệnh nhân phục hình tháo lắp, gồm hàm giả toàn bộ và từng phần.','I1','Denturists are dental providers','directly to the patient.'),
        ('NIDCR tập trung nghiên cứu các khoa học về sức khỏe răng miệng.','I1','The NIH has twenty-seven','the oral health sciences')])
    add('zayan_education','a2v2_82eb724303a753b3','en',
        'Những trải nghiệm nha khoa thời nhỏ giúp Dr. Zayan hiểu điều gì về công việc và phục vụ công chúng?',
        'Sau khi tốt nghiệp bằng BS, Dr. Zayan dạy bao lâu trước khi nhận thấy cần học bằng cao hơn để theo đuổi học thuật?',
        'Trải nghiệm nha khoa thời nhỏ giúp Dr. Zayan hiểu điều gì, và sau bằng BS cô dạy bao lâu trước khi nhận thấy cần học bằng cao hơn?',[
        ('Các trải nghiệm giúp cô hiểu nhân viên phòng nha và lợi ích của việc phục vụ công chúng, cung cấp giáo dục.','I1','These experiences allowed me','the public and providing education.'),
        ('Cô dạy một năm sau bằng BS rồi nhận thấy cần bằng cao hơn nếu muốn theo đuổi nghề học thuật.','I1','After graduating with a BS degree','pursue a career in academia.')])
    add('clinical_research_skills','a2v2_c65eae81ec32e402','en',
        'Theo tài liệu, quản lý và điều phối thử nghiệm lâm sàng đòi hỏi kỹ năng gì?',
        'Ngoài quy định của nhà tài trợ và chính phủ, người làm thử nghiệm lâm sàng còn phải tuân thủ trách nhiệm nào?',
        'Quản lý thử nghiệm lâm sàng đòi hỏi kỹ năng gì, và ngoài quy định của nhà tài trợ/chính phủ còn phải tuân thủ trách nhiệm nào?',[
        ('Quản lý và điều phối thử nghiệm lâm sàng đòi hỏi nhiều kỹ năng tổ chức.','I1','Learning how to manage','organizational skill.'),
        ('Còn phải tuân thủ các trách nhiệm chuyên môn và đạo đức.','I1','In addition to sponsor','must be adhered to.')])
    add('lesson_literature','a2v2_48294a6fcafe8c93','en',
        'Theo bảng thuật ngữ, lesson plan là loại tài liệu dùng cho việc gì?',
        'Theo bảng thuật ngữ, literature review xem xét những tài liệu nào để xác định điều gì?',
        'Theo bảng thuật ngữ, lesson plan dùng cho việc gì và literature review xem xét tài liệu để xác định điều gì?',[
        ('Lesson plan là tài liệu viết dùng để lập kế hoạch cho một bài trình bày.','I1','Lesson plan Written','planning a presentation.'),
        ('Literature review xem xét các báo cáo và nghiên cứu liên quan để xác định điều đã biết về một chủ đề hoặc vấn đề.','I1','Literature review Examination','a specific topic or issue.')])
    add('qualitative_quantitative','a2v2_ba779550f4fc1075','en',
        'Theo bảng thuật ngữ, đánh giá định tính trả lời những câu hỏi nào về chương trình hoặc dự án nghiên cứu?',
        'Theo bảng thuật ngữ, đánh giá định lượng đánh giá chương trình hoặc dự án theo cách nào?',
        'Theo bảng thuật ngữ, đánh giá định tính và đánh giá định lượng khác nhau ở cách đánh giá chương trình như thế nào?',[
        ('Đánh giá định tính trả lời câu hỏi vì sao và bằng cách nào về chương trình y tế công cộng nha khoa hoặc dự án nghiên cứu.','I2','Qualitative evaluation Answering','or research project.'),
        ('Đánh giá định lượng là đánh giá bằng số liệu chương trình y tế công cộng nha khoa hoặc dự án nghiên cứu.','I2','Quantitative evaluation A numerical','or research project.')])
    add('program_baseline','a2v2_aaa84243fb0db221','en',
        'Theo tài liệu, người lập kế hoạch cần dùng kỹ thuật khoa học để xác định điều gì trong và sau chương trình?',
        'Theo tài liệu, dữ liệu nền được thu thập khi nào và có thể dùng để làm gì?',
        'Theo tài liệu, kỹ thuật khoa học giúp xác định điều gì về chương trình, còn dữ liệu nền được thu thập khi nào và dùng làm gì?',[
        ('Kỹ thuật khoa học được dùng để xác định tính hiệu quả trong quá trình và khi hoàn thành chương trình.','I1','During a dental public health program','ascertain effectiveness.'),
        ('Dữ liệu nền được thu thập ở giai đoạn đánh giá trước khi chương trình bắt đầu, có thể dùng để xác định nhu cầu và đánh giá kết quả sau đó.','I2','Baseline data refers','evaluate program outcome.')])
    add('survey_methods','a2v2_d89548e40735d3e4','en',
        'Theo bảng mô tả, BRFSS thu thập dữ liệu sức khỏe bằng phương thức nào?',
        'Theo bảng mô tả, NHANES kết hợp hai phương thức đánh giá nào?',
        'Theo bảng mô tả, BRFSS dùng phương thức nào và NHANES kết hợp hai phương thức nào?',[
        ('BRFSS là khảo sát sức khỏe liên tục được tiến hành qua điện thoại.','I2','Behavioral Risk Factor Surveillance System BRFSS','conducted by telephone.'),
        ('NHANES kết hợp phỏng vấn và khám thực thể.','I2','National Health and Nutrition Examination Survey NHANES','interview and physical examinations')])
    add('research_confidentiality','a2v2_be1bffdb65b25b19','en',
        'Theo đoạn nguồn, danh tính người tham gia được xử lý thế nào khi công bố nghiên cứu?',
        'Theo đoạn nguồn, nhà nghiên cứu có thể xin sự đồng ý của người tham gia để sử dụng gì trong bài báo?',
        'Khi công bố nghiên cứu, danh tính người tham gia được xử lý thế nào và nhà nghiên cứu có thể xin phép sử dụng gì trong bài báo?',[
        ('Khi nghiên cứu được công bố, người tham gia không được định danh.','I1','Research subjects are never','when research is published.'),
        ('Nhà nghiên cứu có thể xin sự đồng ý của người tham gia để sử dụng ảnh trong bài báo.','I1','However, the researcher may ask','photographs in an article.')])

    add('central_relation','a2v2_fda5fe6523acb558','vi',
        'Theo đoạn nguồn, tương quan trung tâm là tương quan giữa những cấu trúc nào?',
        'Theo đoạn nguồn, có tìm đạt tương quan trung tâm khi hệ cơ hàm đang căng, co cứng không?',
        'Theo đoạn nguồn, tương quan trung tâm là tương quan giữa những cấu trúc nào và có tìm đạt khi hệ cơ hàm căng, co cứng không?',[
        ('Tương quan trung tâm là tương quan hàm-sọ, gần hơn là giữa lồi cầu và hõm khớp, qua trung gian đĩa khớp.','I1','Tương quan trung tâm là một tương quan hàm-sọ','cân xứng trên đường giữa.'),
        ('Không thể tìm đạt tương quan trung tâm khi hệ thống cơ hàm đang căng, co cứng.','I1','Không thể tìm đạt TQTT','căng, co cứng;')])
    add('border_contact_movements','a2v2_1cf11ca910119700','vi',
        'Theo tài liệu, vận động biên của hàm dưới được định nghĩa thế nào?',
        'Theo tài liệu, vận động tiếp xúc có đặc điểm gì về trạng thái các răng?',
        'Theo tài liệu, vận động biên và vận động tiếp xúc được phân biệt bằng những đặc điểm nào?',[
        ('Vận động biên đưa hàm dưới tới những vị trí tối đa mà nó có thể thực hiện được.','I2','Vận động biên:','nó có thể thực hiện được.'),
        ('Trong vận động tiếp xúc, răng hàm trên và hàm dưới giữ tiếp xúc; răng dưới trượt trên răng trên nên còn gọi là vận động trượt.','I2','Vận động tiếp xúc:','còn gọi là vận động trượt.')])
    add('splint_materials','a2v2_1e4353fc12a9ba76','vi',
        'Theo đoạn sách, vật liệu làm máng nhai được mô tả là ưa chuộng nhất là gì?',
        'Theo đoạn sách, máng nhai kim loại có ưu điểm gì về độ dày và độ bền hình dạng?',
        'Theo đoạn sách, vật liệu máng nhai được ưa chuộng nhất là gì và máng kim loại có ưu điểm gì về độ dày, độ bền hình dạng?',[
        ('Đoạn sách nêu vật liệu được ưa chuộng nhất là nhựa nấu acrylic trong suốt.','I1','Vật liệu được dùng làm máng nhai','nhựa nấu aerylic trong suốt.'),
        ('Máng nhai kim loại có thể làm mỏng và khó bị gãy hoặc biến dạng.','I1','Ưu điểm của máng nhai kim loại','bị gãy hoặc biến đạng.')])
    add('occlusal_indications','a2v2_c1bed0ae1ef63cec','vi',
        'Theo mục chỉ định, loại tiếp xúc nào tại tương quan trung tâm là chỉ định mài điều chỉnh?',
        'Theo mục chỉ định, loại tiếp xúc nào tại vị trí lồng múi tối đa là chỉ định mài điều chỉnh?',
        'Theo mục chỉ định, tiếp xúc tại tương quan trung tâm và tại lồng múi tối đa được nêu khác nhau thế nào?',[
        ('Tại tương quan trung tâm, chỉ định được nêu là có tiếp xúc sớm.','I1','1. Có tiếp xúc sớm','tương quan trung tâm (TQTT),'),
        ('Tại lồng múi tối đa, chỉ định được nêu là có tiếp xúc quá mức.','I1','2. Có tiếp xúc quá mức','lồng múi tối đa (LMTĐ),')])
    add('closing_arc_limits','a2v2_d0d93585822c12bb','vi',
        'Theo tài liệu, kích thước nhỏ của giá khớp đơn giản dẫn đến khác biệt nào về trục và cung đóng?',
        'Theo tài liệu, có thể dùng sáp cắn liên hàm ở tương quan trung tâm để lên mẫu hàm trên giá khớp đơn giản và thực hiện phục hình không?',
        'Theo tài liệu, giá khớp đơn giản khác bệnh nhân về trục/cung đóng như thế nào và có thể dùng sáp cắn tương quan trung tâm để lên mẫu, làm phục hình không?',[
        ('Do kích thước nhỏ, trục bản lề của giá khớp không trùng trục của bệnh nhân và cung đóng khác cung đóng của bệnh nhân.','I2','Như trên đã phân tích','khác cung đóng của bệnh nhân (Hình 10-4).'),
        ('Đoạn nguồn nói không thể dùng sáp cắn liên hàm ở tương quan trung tâm để lên mẫu trên giá khớp đơn giản và làm phục hình, vì quan hệ mẫu hàm không giống trên miệng.','I2','Người ta không thể sử dụng sáp cắn liên hàm','không giống quan hệ đó trên miệng.')])
    add('supporting_cusps','a2v2_836c5246b30f8d52','vi',
        'Theo tài liệu, múi gần trong của răng cối lớn trên ăn khớp với vị trí nào ở hàm dưới?',
        'Theo tài liệu, múi trong của răng cối nhỏ trên ăn khớp với vị trí nào ở hàm dưới?',
        'Theo tài liệu, múi gần trong răng cối lớn trên và múi trong răng cối nhỏ trên lần lượt ăn khớp với vị trí nào ở hàm dưới?',[
        ('Múi gần trong răng cối lớn trên ăn khớp với trũng giữa răng cùng tên hàm dưới.','I2','Các múi gần trong răng cối lớn trên','tạo thành ba điểm tiếp xúc.'),
        ('Múi trong răng cối nhỏ trên ăn khớp với trũng tam giác xa của răng cùng tên hàm dưới.','I2','Các múi trong răng cối nhỏ trên','tạo thành hai đến ba điểm tiếp xúc.')])
    add('splint_borders','a2v2_e3a730b3f2730b4d','vi',
        'Theo tài liệu, giới hạn phía trong máng nhai phải theo đường viền nào và tránh chèn ép vùng nào?',
        'Theo tài liệu, bờ khẩu cái của máng nhai phải liên hệ với các gờ niêm mạc như thế nào?',
        'Theo tài liệu, giới hạn phía trong và bờ khẩu cái của máng nhai cần đáp ứng những yêu cầu nào?',[
        ('Giới hạn phía trong theo đường viền giải phẫu của nướu, không chèn ép bờ nướu rời.','I2','Giới hạn phía trong cần','bờ nướu rời (Hình 12-5).'),
        ('Bờ khẩu cái phải phù hợp với gờ niêm mạc khẩu cái và không bắt chéo các gờ.','I2','Bờ khẩu cái phải phù hợp','không bắt chéo những gờ này.')])
    add('flexors_extensors','a2v2_45fb472076f0b7f0','vi',
        'Theo tài liệu, nhóm cơ nào thực hiện cử động làm hai xương gần lại nhau?',
        'Theo tài liệu, nhóm cơ nào thực hiện cử động làm hai xương xa nhau?',
        'Theo tài liệu, cử động làm hai xương gần lại và xa nhau lần lượt do nhóm cơ nào thực hiện?',[
        ('Các cơ gấp thực hiện cử động làm hai xương gần lại nhau.','I1','Cử động đó có thể làm hai xương gần lại nhau','thực hiện bởi các cơ gấp'),
        ('Các cơ duỗi thực hiện cử động làm hai xương xa nhau.','I1','hoặc xa nhau (duỗi)','thực hiện bởi các cơ duỗi')])
    add('articular_disc','a2v2_82892510fdde1f82','vi',
        'Theo đoạn nguồn, đĩa khớp được cấu tạo từ mô sợi hay mô sụn?',
        'Theo đoạn nguồn, chu vi đĩa liên tục với mô nào và chia khoang khớp thành bao nhiêu buồng?',
        'Theo đoạn nguồn, đĩa khớp được cấu tạo từ loại mô nào và chu vi đĩa liên tục với mô gì để chia khoang khớp thành mấy buồng?',[
        ('Đĩa khớp được cấu tạo từ mô sợi keo săn chắc, không phải mô sụn.','I1','Đĩa khớp được cấu tạo từ mô sợi keo','chứ không phải là mô sụn'),
        ('Chu vi đĩa liên tục với mô liên kết quanh đĩa và chia khoang khớp thành hai buồng.','I2','Toàn bộ chu vi của đĩa','thành hai buồng (xem phần bao khớp).')])
    add('quick_master','a2v2_674a4c6419998d2d','vi',
        'Theo thông số Quick-Master trong tài liệu, mặt phẳng tham chiếu là mặt phẳng nào?',
        'Theo thông số Quick-Master trong tài liệu, khoảng cách hai lồi cầu là bao nhiêu?',
        'Theo tài liệu, Quick-Master dùng mặt phẳng tham chiếu nào và có khoảng cách hai lồi cầu bao nhiêu?',[
        ('Mặt phẳng tham chiếu là mặt phẳng Frankfort.','I1','Mặt phẳng tham chiếu:','mặt phẳng Frankfort (Hình 10-18).'),
        ('Khoảng cách hai lồi cầu là 110 mm.','I1','Khoảng cách hai lỗi cầu:','110mm.')])
    add('opening_movements','a2v2_50ad7cd69a168e27','vi',
        'Khi điểm răng cửa vạch đoạn BH theo tài liệu, lồi cầu dịch chuyển theo hướng nào?',
        'Theo tài liệu, trong giai đoạn đó trục quay hàm dưới dịch chuyển và nằm gần vị trí nào?',
        'Khi điểm răng cửa vạch đoạn BH, lồi cầu dịch chuyển thế nào và trục quay hàm dưới di chuyển, nằm gần vị trí nào?',[
        ('Lồi cầu rời hõm khớp, dịch chuyển ra trước và xuống dưới.','I1','Khi điểm răng cửa vạch đoạn BH,','dịch chuyển ra trước và xuống dưới,'),
        ('Trục quay rời lồi cầu, di chuyển xuống dưới và ra trước, nằm gần phía dưới lỗ hàm.','I1','Trục quay của vận động hàm dưới','nằm gần phía dưới lỗ hàm.')])
    add('functional_occlusion','a2v2_78427079dcd09e09','vi',
        'Theo tiêu chuẩn được nêu, các thành phần hệ thống nhai cần hài hòa về những mặt nào?',
        'Theo tiêu chuẩn được nêu, vận động hàm dưới được mô tả thế nào về sự trơn tru và cản trở?',
        'Theo tài liệu, tiêu chuẩn khớp cắn sinh lý mô tả sự hài hòa của hệ thống nhai và vận động hàm dưới như thế nào?',[
        ('Các thành phần hệ thống nhai hài hòa về hình thái và chức năng, góp phần ổn định khớp cắn.','I1','Các thành phần của hệ thống nhai hài hòa','góp phần ổn định khớp cắn;'),
        ('Hàm dưới vận động dễ dàng, trơn tru, không có cản trở cắn khớp trong các vận động trượt.','I1','Hàm dưới vận động dễ dàng','các vận động trượt của hàm dưới.')])
    add('passive_movement','a2v2_8897d8fe42290a8d','vi',
        'Theo đoạn nguồn, vận động thụ động được thực hiện thế nào khi phân biệt giới hạn do cơ hay khớp?',
        'Theo đoạn nguồn, đau khi vận động chủ động nhưng không đau khi vận động thụ động gợi ý nguyên nhân nào?',
        'Theo đoạn nguồn, vận động thụ động được thực hiện ra sao và đau chủ động nhưng không đau thụ động gợi ý nguyên nhân nào?',[
        ('Thầy thuốc hướng dẫn hàm dưới trong khi các cơ thư giãn để đánh giá vận động thụ động.','I2','Để phân biệt giới hạn vận động','(thực hiện vận động bản lề).'),
        ('Có thể nghĩ đến nguyên nhân do cơ khi vận động chủ động đau nhưng vận động thụ động không đau.','I2','Đây không phải là một test đặc hiệu','nguyên nhân gây đau là do cơ.')])
    add('soft_hard_end_feel','a2v2_550d32c2c7ffe7e9','vi',
        'Theo mô tả test end feel, cảm giác kết thúc mềm có đặc điểm và gợi ý nguyên nhân gì?',
        'Theo mô tả test end feel, cảm giác kết thúc cứng có đặc điểm và gợi ý nguyên nhân gì?',
        'Theo tài liệu, cảm giác kết thúc mềm và cứng khác nhau thế nào về mức tăng độ há và nguyên nhân giới hạn?',[
        ('Kết thúc mềm là khi độ há tăng được thêm, có thể kèm đau; giới hạn do cơ hoặc các mô mềm khác.','I2','Nếu độ há miệng tăng được thêm','nguyên nhân ở cơ hay các mô mềm khác.'),
        ('Kết thúc cứng là khi độ há không tăng thêm được; giới hạn do nguyên nhân ở khớp như sai vị trí đĩa hoặc bất thường xương.','I2','Nếu độ há miệng không thể tăng thêm được:','những bất thường về cấu trúc xương.')])
    add('bonwill','a2v2_6884732c6805b2e9','vi',
        'Theo tài liệu, Bonwill trình bày những quy luật hình học và cơ học của sự ăn khớp răng vào năm nào?',
        'Theo tài liệu, Bonwill mô tả hàm dưới bằng hình gì, kích thước bao nhiêu và các đỉnh ở đâu?',
        'Theo tài liệu, Bonwill trình bày quy luật ăn khớp răng năm nào và mô tả hàm dưới bằng hình, kích thước, các đỉnh như thế nào?',[
        ('Bonwill trình bày những quy luật hình học và cơ học của sự ăn khớp răng năm 1855.','I3','Năm 1855 Bonwill,','sự ăn khớp các răng”.'),
        ('Ông mô tả hàm dưới như tam giác đều 10 cm, có ba đỉnh là hai lồi cầu và góc cắn gần răng cửa giữa dưới.','I2','Bonwill phân tích và mô tả','răng cửa giữa dưới (Hình 3-20).')])
    add('functional_parafunctional','a2v2_ec07350932c3a0f0','vi',
        'Theo bảng thuật ngữ, hoạt động chức năng của hệ thống nhai nhằm mục đích gì?',
        'Theo bảng thuật ngữ, hoạt động cận chức năng có đặc điểm gì về ý thức, sự lặp lại và mục đích?',
        'Theo bảng thuật ngữ, hoạt động chức năng và cận chức năng của hệ thống nhai khác nhau như thế nào?',[
        ('Hoạt động chức năng là hoạt động của hệ thống nhai nhằm thực hiện chức năng.','I1','Hoạt động chúc năng:','nhằm thực hiện chức năng.'),
        ('Hoạt động cận chức năng có thể có hoặc không có ý thức, lặp đi lặp lại, có hoặc không tiếp xúc răng và không nhằm thực hiện chức năng.','I1','Hoạt động cận chức năng:','không nhằm thực hiện chức năng.')])
    assert len(result)==32
    return result
