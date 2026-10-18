"""Giữ ảnh gốc của các bảng PMC không có ô văn bản trong XML."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import parse_qs,urlsplit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.bulk_public import OUTPUT,RAW,S3_BASE,HttpClient,load_documents,load_json,save_json
from src.data.public_schema import digest,now_utc


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    client=HttpClient()
    assets=[];errors=[]
    try:
        for doc in load_documents(OUTPUT/'pmc_documents.jsonl'):
            blocks=[block for section in doc.sections for block in section.blocks if 'table_structure_requires_review' in block.quality_flags]
            if not blocks:
                continue
            receipt=load_json(RAW/'pmc/cloud'/(doc.document_id+'.receipt.json'),{})
            media=receipt.get('metadata',{}).get('media_urls',[])
            root=ET.fromstring(Path(doc.raw.path).read_bytes())
            for node in root.iter():
                node.tag=node.tag.rsplit('}',1)[-1]
            article=root if root.tag=='article' else root.find('.//article')
            for block in blocks:
                node=article.find(block.locators[0].path.removeprefix('/article/'))
                for graphic in node.findall('.//graphic'):
                    name=graphic.get('{http://www.w3.org/1999/xlink}href','')
                    match=next((url for url in media if Path(urlsplit(url).path).name==Path(name).name),None)
                    if not match:
                        errors.append({'document_id':doc.document_id,'block_id':block.block_id,'error':'Không có ảnh trong metadata PMC Cloud','filename':name})
                        continue
                    parsed=urlsplit(match)
                    if parsed.scheme!='s3' or parsed.netloc!='pmc-oa-opendata':
                        raise ValueError('Ảnh không thuộc PMC Cloud Service')
                    url=S3_BASE+parsed.path.lstrip('/')+('?' + parsed.query if parsed.query else '')
                    expected=parse_qs(parsed.query).get('md5',[None])[0]
                    path=RAW/'pmc/media'/doc.document_id/Path(name).name
                    if path.exists():
                        data=path.read_bytes()
                    else:
                        data=client.get(url).content
                    if expected and hashlib.md5(data).hexdigest()!=expected:
                        raise ValueError('Ảnh bảng không khớp MD5 PMC')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    if not path.exists():
                        path.write_bytes(data)
                    assets.append({'document_id':doc.document_id,'document_revision':doc.revision_id,
                        'block_id':block.block_id,'filename':name,'path':str(path),'sha256':digest(data),
                        'source_url':url,'source_locator':block.locators[0].model_dump(),
                        'status':'image_saved_not_ocr','checked_at':now_utc()})
                    print(f'{doc.document_id}: đã lưu {name}',flush=True)
    finally:
        client.session.close()
    path=OUTPUT/'table_images.jsonl'
    path.write_text(''.join(json.dumps(asset,ensure_ascii=False)+'\n' for asset in assets),encoding='utf-8')
    save_json(OUTPUT/'table_images_collection.json',{'assets':len(assets),'errors':errors,'checked_at':now_utc()})
    print(f'Ảnh bảng: {len(assets)}, lỗi: {len(errors)}')
    return 1 if errors else 0


if __name__=='__main__':
    raise SystemExit(main())
