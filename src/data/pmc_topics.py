"""Chủ đề và câu truy vấn dùng để chọn bài PMC cho kho công khai."""

PMC_TOPICS = [
    {"id": "caries", "name": "Sâu răng", "query": 'TITLE_ABS:"dental caries" OR TITLE_ABS:"tooth decay"'},
    {"id": "endodontics", "name": "Bệnh tủy và quanh chóp", "query": 'TITLE_ABS:pulpitis OR TITLE_ABS:"apical periodontitis" OR TITLE_ABS:"root canal treatment"'},
    {"id": "gingivitis", "name": "Viêm lợi", "query": "TITLE_ABS:gingivitis"},
    {"id": "periodontitis", "name": "Viêm nha chu", "query": "TITLE_ABS:periodontitis"},
    {"id": "odontogenic_infection", "name": "Nhiễm trùng và áp xe răng", "query": 'TITLE_ABS:"odontogenic infection" OR TITLE_ABS:"dental abscess" OR TITLE_ABS:"periapical abscess"'},
    {"id": "third_molar", "name": "Răng khôn và viêm quanh thân răng", "query": 'TITLE_ABS:"third molar" OR TITLE_ABS:pericoronitis OR TITLE_ABS:"wisdom teeth"'},
    {"id": "oral_mucosa", "name": "Bệnh niêm mạc miệng", "query": 'TITLE_ABS:"oral lichen planus" OR TITLE_ABS:"oral leukoplakia" OR TITLE_ABS:"recurrent aphthous" OR TITLE_ABS:"oral candidiasis"'},
    {"id": "xerostomia", "name": "Khô miệng và tuyến nước bọt", "query": "TITLE_ABS:xerostomia OR TITLE_ABS:hyposalivation"},
    {"id": "halitosis", "name": "Hôi miệng", "query": "TITLE_ABS:halitosis"},
    {"id": "dental_trauma", "name": "Chấn thương răng", "query": 'TITLE_ABS:"dental trauma" OR TITLE_ABS:"tooth avulsion" OR TITLE_ABS:"traumatic dental injuries"'},
    {"id": "tmd", "name": "Rối loạn khớp thái dương hàm", "query": 'TITLE_ABS:"temporomandibular disorder" OR TITLE_ABS:"temporomandibular disorders"'},
    {"id": "tooth_wear", "name": "Mòn răng và ê buốt ngà", "query": 'TITLE_ABS:"tooth wear" OR TITLE_ABS:"dental erosion" OR TITLE_ABS:"dentin hypersensitivity"'},
]
