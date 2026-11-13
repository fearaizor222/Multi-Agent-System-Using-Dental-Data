"""Stage 2 (v4) audited OCR/layout source cleaner for dental RAG chunks and seeds.

Keeps runtime protocol normalization (`normalize_source_content`) separate from
corpus typo fixes so both Stage 2 dataset construction and gold span lookup stay
100% deterministic and source-verified.
"""
from __future__ import annotations

from copy import deepcopy
import re

from src.training.a2_sentence_protocol import normalize_source_content

# Ordered longest/most-specific phrases first to prevent partial substitutions.
OCR_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ('c222772227>:72/2//00 “2 "2 tược? lon”, 5/0} Hình 12-5.', 'Hình 12-5.'),
    ('Hãm khớp thúi dương hàm: maột vòm', 'Hõm khớp thái dương hàm: một vòm'),
    ('tối tứu -hJ_ nhều', 'tối ưu'),
    ('ở. Có cần trở', '3. Có cản trở'),
    ('mồm tháp xương khẩu eái', 'mỏm tháp xương khẩu cái'),
    ('địch của mô hoạt dịch', 'dịch của mô hoạt dịch'),
    ('THỜI GIAN ĐIỀU TRỊ | Thời gian', 'THỜI GIAN ĐIỀU TRỊ\nThời gian'),
    ('sphenomenicus muscle).', 'sphenomenicus muscle)).'),
    ('Cơ chân bướm rong', 'Cơ chân bướm trong'),
    ('Progressive side shif†', 'Progressive side shift'),
    ('mmediate side shift', 'immediate side shift'),
    ('vertteal đimension', 'vertical dimension'),
    ('Vận động bản lễ', 'Vận động bản lề'),
    ('đường cong Snpee', 'đường cong Spee'),
    ('cận chức ()_', 'cận chức'),
    ('//ên thuộc nhau', 'liên thuộc nhau'),
    ('0 100 | ị', '0 100'),
    ('Diên Thảo ()', 'Diên Thảo'),
    (' . () ', '. '),
    ('. () ', '. '),
    (' () ', ' '),
    ('răng cửa đưới', 'răng cửa dưới'),
    ('mặt phẳng hgang', 'mặt phẳng ngang'),
    ('Quiek-Master', 'Quick-Master'),
    ('Type Ii Error', 'Type II Error'),
    ('Băng Hàm Mặt', 'Răng Hàm Mặt'),
    ('Có cần trở', 'Có cản trở'),
    ('cấu trúc lôi', 'cấu trúc lồi'),
    ('điện cơ đổ', 'điện cơ đồ'),
    ('lỗổng múi', 'lồng múi'),
    ('thới dương', 'thái dương'),
    ('thái dướng', 'thái dương'),
    ('thúi dương', 'thái dương'),
    ('tham chiêu', 'tham chiếu'),
    ('đường viển', 'đường viền'),
    ('nhai nghiên', 'nhai nghiền'),
    ('dân dần', 'dần dần'),
    ('lâu đài', 'lâu dài'),
    ('trục đài', 'trục dài'),
    ('chiều đài', 'chiều dài'),
    ('nhân.để', 'nhân để'),
    ('một. cách', 'một cách'),
    ('ĐỀ LÊN', 'ĐỂ LÊN'),
    ('MỤC TIỂU', 'MỤC TIÊU'),
    ('BẢN LỄ', 'BẢN LỀ'),
    ('KHỚP CĂN', 'KHỚP CẮN'),
    ('há thối đa', 'há tối đa'),
    ('khuyếch tán', 'khuếch tán'),
    ('Dịch bhớp', 'Dịch khớp'),
    ('địch chuyển', 'dịch chuyển'),
    ('hoạt địch', 'hoạt dịch'),
    ('thường mông', 'thường mỏng'),
    ('thây thuốc', 'thầy thuốc'),
    ('chúc năng', 'chức năng'),
    ('hình đáng', 'hình dáng'),
    ('uận động', 'vận động'),
    ('điện khớp', 'diện khớp'),
    ('trơn nhắn', 'trơn nhẵn'),
    ('cần trở', 'cản trở'),
    ('biến đạng', 'biến dạng'),
    ('Hãm khớp', 'Hõm khớp'),
    ('của khóp', 'của khớp'),
    ('lỗi khớp', 'lồi khớp'),
    ('lồi khóp', 'lồi khớp'),
    ('lỗi cầu', 'lồi cầu'),
    ('lỗi câu', 'lồi cầu'),
    ('lổi cầu', 'lồi cầu'),
    ('lôi cầu', 'lồi cầu'),
    ('lôi câu', 'lồi cầu'),
    ('Khi li cầu', 'Khi lồi cầu'),
    ('chức năng lối %u', 'chức năng tối ưu'),
    ('những điêu tiên quyết', 'những điều tiên quyết'),
    ('lén giá khớp', 'lên giá khớp'),
    ('tức gần lễ hàm', 'tức gần lỗ hàm'),
    ('2,34 mm + 0,93) Ở).', '2,34 mm + 0,93).'),
    ('lỗi củ', 'lồi củ'),
    ('hàm đưới', 'hàm dưới'),
    ('răng đưới', 'răng dưới'),
    ('lỗng múi', 'lồng múi'),
    ('tối tứu', 'tối ưu'),
    ('căng dãn', 'căng giãn'),
    ('thư dãn', 'thư giãn'),
    ('đây chằng', 'dây chằng'),
    ('khí nhai', 'khi nhai'),
    ('hơi lôi', 'hơi lồi'),
    ('cắm xúc', 'cảm xúc'),
    ('bản lễ', 'bản lề'),
    ('bản lâ', 'bản lề'),
    ('Bót trên', 'Bó trên'),
    ('bỏ trân', 'bó trên'),
    ('động tắc', 'động tác'),
    ('aerylic', 'acrylic'),
    ('Snpee', 'Spee'),
    ('TQTTD', 'TQTT'),
    ('TƠTIT', 'TQTT'),
    ('TQTTT', 'TQTT'),
    ('Hòang', 'Hoàng'),
    ('đễ phạm', 'dễ phạm'),
    ('Độ đốc', 'Độ dốc'),
    ('11.Ôc', '11. Ốc'),
    ('điễn ra', 'diễn ra'),
    ('tên tại', 'tồn tại'),
    ('maột vòm', 'một vòm'),
    ('leeuay', 'leeway'),
    ('ghỉ nhận', 'ghi nhận'),
    ('đuy trì', 'duy trì'),
    ('Ramjford', 'Ramfjord'),
    ('Ramfiord', 'Ramfjord'),
    ('thay đối', 'thay đổi'),
    ('tác đụng', 'tác dụng'),
    ('điểu này', 'điều này'),
    ('đàn hải', 'đàn hồi'),
    ('ít nhiễu', 'ít nhiều'),
    ('Rhi tất cả', 'Khi tất cả'),
    ('Ehi há', 'Khi há'),
    ('xuơng', 'xương'),
    ('kể mặt', 'bề mặt'),
    ('đời về', 'dời về'),
    ('yếu tế', 'yếu tố'),
    ('dữ đội', 'dữ dội'),
    ('Cá hai', 'Cả hai'),
    ('lực eo', 'lực co'),
    ('1.1.ã.', '1.1.5.'),
    ('LMT?Đ', 'LMTĐ'),
    ('VỊ trí', 'Vị trí'),
    ('shif†', 'shift'),
    ('seale', 'scale'),
    ('đưới', 'dưới'),
    ('đải', 'dải'),
    ('uùà', 'và'),
    ('eơ', 'cơ'),
    ('Top —. ', ''),
    ('ọ › ', ' '),
    ('ự ⁄ ', ''),
    ('B=)', ''),
)


def clean_source_text(text: str) -> str:
    """Apply runtime unicode/bullet normalization plus audited OCR typo fixes."""
    cleaned = normalize_source_content(text)
    for old, new in OCR_REPLACEMENTS:
        cleaned = cleaned.replace(old, new)
    cleaned = re.sub(r'[ ]{2,}', ' ', cleaned)
    cleaned = re.sub(r' *\n *', '\n', cleaned)
    return cleaned.strip()


def clean_hard_source(source: dict) -> dict:
    """Return a cleaned copy of an `a2_hard_v3_sources` chunk dict."""
    updated = deepcopy(source)
    updated['content'] = clean_source_text(source['content'])
    if 'title' in updated and isinstance(updated['title'], str):
        updated['title'] = normalize_source_content(updated['title'])
    return updated


def clean_seed_record(seed: dict) -> dict:
    """Clean sources and re-anchor claim quotes on a `seed_review` family record."""
    updated = deepcopy(seed)
    for src in updated['sources'].values():
        src['content'] = clean_source_text(src['content'])
        if 'title' in src and isinstance(src['title'], str):
            src['title'] = normalize_source_content(src['title'])

    for claim in updated.get('claims', []):
        src_text = updated['sources'][claim['source_key']]['content']
        quote = clean_source_text(claim['quote'])
        start = src_text.find(quote)
        if start < 0 or src_text.find(quote, start + 1) >= 0:
            raise ValueError(
                f"Failed to uniquely anchor cleaned quote for seed={updated['id']}: {quote!r}"
            )
        claim['quote'] = quote
        claim['quote_start'] = start
        claim['quote_end'] = start + len(quote)
    return updated
