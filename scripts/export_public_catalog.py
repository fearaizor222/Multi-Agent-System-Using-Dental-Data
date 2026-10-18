"""Xuất danh sách có tìm kiếm và bản đọc từng tài liệu để rà soát cục bộ."""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.bulk_public import OUTPUT,ROOT,load_documents,load_json
from src.data.public_catalog import render_document
from src.data.pmc_topics import PMC_TOPICS


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    target=ROOT/'outputs/public_v1_bulk'
    documents=[doc for source in ('byt','msd','pmc') for doc in load_documents(OUTPUT/(source+'_documents.jsonl'))]
    checks=load_json(target/'checks.json',{})
    if not checks.get('passed'):
        raise ValueError('Cần chạy check_public_bulk thành công trước khi xuất danh sách')
    folder=target/'documents'
    folder.mkdir(parents=True,exist_ok=True)
    style='''<style>body{font:16px/1.6 Arial;color:#222;max-width:1250px;margin:28px auto;padding:0 20px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #bbb;padding:8px;text-align:left;vertical-align:top}th{background:#eef1f5}small{color:#666;font-size:12px;overflow-wrap:anywhere}.block{margin:16px 0}input,select{font:inherit;padding:8px;margin:8px 12px 16px 0}p{white-space:pre-wrap}a{color:#145ca8}h2,h3,h4{margin-top:24px}</style>'''
    header='<!doctype html><html lang="vi"><meta charset="utf-8">'
    escape=html.escape
    assets=[]
    asset_file=OUTPUT/'table_images.jsonl'
    if asset_file.exists():
        assets=[json.loads(line) for line in asset_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    topics={item['id']:item['name'] for item in PMC_TOPICS}
    rows=[]
    for doc in documents:
        output=folder/(doc.document_id+'.html')
        content=render_document(doc)
        images=[asset for asset in assets if asset['document_id']==doc.document_id]
        if images:
            content+='<h2>Bảng gốc dạng ảnh</h2>'
            for asset in images:
                name=asset['sha256']+Path(asset['filename']).suffix
                target_image=folder/'media'/name
                target_image.parent.mkdir(parents=True,exist_ok=True)
                target_image.write_bytes(Path(asset['path']).read_bytes())
                content+=f'<p>{escape(asset["source_locator"].get("source_id") or asset["filename"])}</p><img style="max-width:100%" src="media/{name}" alt="Bảng gốc từ PMC">'
        output.write_text(header+'<title>'+escape(doc.title)+'</title>'+style+'<p><a href="../index.html">← Danh sách tài liệu</a></p>'+content+'</html>',encoding='utf-8')
        title=escape(doc.title)
        topic=topics.get(doc.source_metadata.get('collection_topic'), 'Hướng dẫn Bộ Y tế' if doc.source=='byt' else 'Nha khoa MSD')
        tables=sum(block.kind=='table' for section in doc.sections for block in section.blocks)
        source='Bộ Y tế' if doc.source=='byt' else doc.source.upper()
        rows.append(f'<tr data-source="{doc.source}"><td>{source}</td><td><a href="documents/{doc.document_id}.html">{title}</a><br><small>{escape(doc.document_id)}</small></td><td>{escape(topic)}</td><td>{doc.language}</td><td>{escape(doc.publication_date or doc.updated_date or "")}</td><td>{tables}</td><td><a href="{escape(doc.source_url,quote=True)}">Nguồn</a></td></tr>')
    page=header+'<title>Kho công khai C20</title>'+style+'<h1>Kho tài liệu công khai</h1>'
    page+=f'<p>{checks["documents"]["msd"]} bài MSD · {checks["documents"]["pmc"]} bài PMC · {checks["byt_chapters"]} chương Bộ Y tế<br>{checks["chunks"]:,} đoạn · tối đa {checks["max_tokens"]} token/đoạn</p>'
    page+='<p>Mở tên bài để xem nội dung đã xử lý, bảng và vị trí nguồn. Danh sách này phục vụ kiểm tra dữ liệu.</p>'
    page+='<label>Tìm bài <input id="search" placeholder="Tên bài, chủ đề, mã tài liệu"></label><select id="source"><option value="">Tất cả nguồn</option><option value="byt">Bộ Y tế</option><option value="msd">MSD</option><option value="pmc">PMC</option></select><span id="count"></span>'
    page+='<table><thead><tr><th>Nguồn</th><th>Tài liệu</th><th>Chủ đề</th><th>Ngôn ngữ</th><th>Ngày</th><th>Bảng</th><th>Link</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
    page+='''<script>const rows=[...document.querySelectorAll('tbody tr')];function plain(t){return t.normalize('NFD').replace(/[\u0300-\u036f]/g,'').replace(/đ/g,'d').toLowerCase()}function filter(){let n=0;const q=plain(document.querySelector('#search').value),s=document.querySelector('#source').value;for(const row of rows){const yes=(!s||row.dataset.source===s)&&plain(row.textContent).includes(q);row.hidden=!yes;n+=yes}document.querySelector('#count').textContent=n+' tài liệu'}document.querySelector('#search').oninput=filter;document.querySelector('#source').onchange=filter;filter();</script></html>'''
    (target/'index.html').write_text(page,encoding='utf-8')
    print(f'Đã xuất {len(documents)} bản đọc\n{target/"index.html"}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
