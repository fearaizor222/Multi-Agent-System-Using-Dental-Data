"""Kiểm tra dữ liệu sau thu thập, chia đoạn và xuất sổ tài liệu để rà soát."""
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter,defaultdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.bulk_public import OUTPUT,ROOT,candidate_reason,title_topic_reason,load_documents,load_json,save_json
from src.data.public_chunks import make_chunks,validate_chunk_sources
from src.data.public_schema import digest,now_utc


def check_text_coverage(doc,chunks):
    spans=defaultdict(list)
    for chunk in chunks:
        for span in chunk.spans:
            spans[span.block_id].append((span.start,span.end))
    for section in doc.sections:
        for block in section.blocks:
            end=0
            for start,stop in sorted(set(spans[block.block_id])):
                if start>end and block.text[end:start].strip():
                    raise ValueError(f'Chunk bỏ sót văn bản trong {block.block_id}')
                end=max(end,stop)
            if block.text[end:].strip():
                raise ValueError(f'Chunk bỏ sót phần cuối {block.block_id}')


def check_source_locators(doc,raw_data):
    locators=[section.locator for section in doc.sections]
    locators += [loc for section in doc.sections for block in section.blocks for loc in block.locators]
    if doc.source=='pmc':
        root=ET.fromstring(raw_data)
        for node in root.iter():
            node.tag=node.tag.rsplit('}',1)[-1]
        article=root if root.tag=='article' else root.find('.//article')
        for loc in locators:
            node=article if loc.path=='/article' else article.find(loc.path.removeprefix('/article/'))
            if node is None or (loc.source_id and node.get('id')!=loc.source_id):
                raise ValueError(f'Vị trí XML không khớp: {loc.path}')
    elif doc.source=='msd':
        from bs4 import BeautifulSoup
        soup=BeautifulSoup(raw_data,'html.parser')
        for loc in locators:
            if not loc.path:
                continue
            node=soup
            for name,index in re.findall(r'/([^/\[]+)\[(\d+)\]',loc.path):
                found=node.find_all(name,recursive=False)
                if len(found)<int(index):
                    raise ValueError(f'Vị trí HTML không khớp: {loc.path}')
                node=found[int(index)-1]
    else:
        if any(not loc.pdf_page or not 1<=loc.pdf_page<=doc.source_metadata['pdf_page_count'] for loc in locators):
            raise ValueError('Vị trí trang PDF không hợp lệ')
    return len(locators)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small',local_files_only=True)
    target=ROOT/'outputs/public_v1_bulk'
    target.mkdir(parents=True,exist_ok=True)
    docs=[doc for source in ('byt','msd','pmc') for doc in load_documents(OUTPUT/(source+'_documents.jsonl'))]
    ids,dois,source_hashes=set(),set(),set()
    counts=Counter();languages=Counter();chunk_counts=Counter();block_kinds=Counter();flags=Counter()
    inventory=[];reviews=[];errors=[];max_tokens=0;chunk_total=0;chunk_ids=set();locator_total=0
    expected_images=set()
    temporary=OUTPUT/'chunks.jsonl.tmp'
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        for doc in docs:
            try:
                if doc.document_id in ids:
                    raise ValueError('Trùng mã tài liệu')
                ids.add(doc.document_id)
                doi=str(doc.source_metadata.get('doi') or '').lower()
                if doi and doi in dois:
                    raise ValueError('Trùng DOI')
                if doi:
                    dois.add(doi)
                raw_data=Path(doc.raw.path).read_bytes()
                if digest(raw_data)!=doc.raw.sha256:
                    raise ValueError('Bản gốc không khớp SHA-256')
                locator_total+=check_source_locators(doc,raw_data)
                archive=OUTPUT/'revisions'/doc.document_id/(doc.revision_id+'.json')
                if not archive.exists():
                    raise ValueError('Thiếu bản tài liệu theo revision')
                if doc.source=='pmc':
                    item={'pmcid':doc.document_id,'title':doc.title,'isOpenAccess':'Y','language':{'en':'eng','vi':'vie'}.get(doc.language),
                          'pubTypeList':{'pubType':doc.source_metadata.get('pubmed_publication_types',[])}}
                    reason=candidate_reason(item) or title_topic_reason(doc.title,doc.source_metadata['collection_topic'])
                    if reason:
                        raise ValueError(reason)
                pieces=make_chunks(doc,tokenizer)
                validate_chunk_sources(doc,pieces)
                check_text_coverage(doc,pieces)
                for chunk in pieces:
                    if chunk.chunk_id in chunk_ids:
                        raise ValueError('Trùng mã chunk')
                    chunk_ids.add(chunk.chunk_id)
                    stream.write(chunk.model_dump_json()+'\n')
                    flags.update(chunk.quality_flags)
                blocks=[block for section in doc.sections for block in section.blocks]
                for block in blocks:
                    review_flags=[flag for flag in block.quality_flags if flag in {'table_structure_requires_review','pdf_table_header_inferred'}]
                    if review_flags:
                        reviews.append({'document_id':doc.document_id,'title':doc.title,'block_id':block.block_id,
                            'flags':'; '.join(review_flags),'source_url':block.locators[0].url,
                            'pdf_pages':'; '.join(str(loc.pdf_page) for loc in block.locators if loc.pdf_page)})
                    if 'table_structure_requires_review' in block.quality_flags:
                        expected_images.add(block.block_id)
                counts[doc.source]+=1;languages[doc.language]+=1;chunk_counts[doc.source]+=len(pieces)
                block_kinds.update(block.kind for block in blocks)
                max_tokens=max(max_tokens,max(chunk.token_count for chunk in pieces))
                chunk_total+=len(pieces)
                source_hashes.add(doc.raw.sha256)
                inventory.append({'document_id':doc.document_id,'source':doc.source,'language':doc.language,'title':doc.title,
                    'topic':doc.source_metadata.get('collection_topic',''),'publication_date':doc.publication_date or '',
                    'source_url':doc.source_url,'doi':doi,'revision_id':doc.revision_id,
                    'sections':len(doc.sections),'blocks':len(blocks),'tables':sum(block.kind=='table' for block in blocks),
                    'chunks':len(pieces),'processing_flags':'; '.join(doc.processing.quality_flags)})
            except (ValueError,OSError,KeyError) as exc:
                errors.append({'document_id':doc.document_id,'error':str(exc)})
            if len(inventory)%25==0:
                print(f'Đã kiểm tra {len(inventory)}/{len(docs)} tài liệu',flush=True)
    msd=load_json(OUTPUT/'msd_collection.json',{})
    pmc=load_json(OUTPUT/'pmc_collection.json',{})
    byt=next((doc for doc in docs if doc.source=='byt'),None)
    chapters=[]
    if byt:
        for section in byt.sections:
            match=re.match(r'^(\d+)\s*\.',section.title)
            if section.parent_section_id is None and match:
                chapters.append(int(match[1]))
    if set(chapters)!=set(range(1,58)) or len(chapters)!=57:
        errors.append({'source':'byt','error':'Không đủ đúng 57 chương'})
    if msd.get('errors') or msd.get('stopped_reason') or counts['msd']!=msd.get('documents'):
        errors.append({'source':'msd','error':'Chưa khớp kết quả rà mục lục'})
    if not pmc.get('complete') or not 200<=counts['pmc']<=500:
        errors.append({'source':'pmc','error':'Chưa hoàn tất phạm vi 200–500 bài'})
    image_path=OUTPUT/'table_images.jsonl'
    assets=[json.loads(line) for line in image_path.read_text(encoding='utf-8').splitlines() if line.strip()] if image_path.exists() else []
    current_versions={doc.document_id:doc.revision_id for doc in docs}
    for asset in assets:
        if (digest(Path(asset['path']).read_bytes())!=asset['sha256']
                or current_versions.get(asset['document_id'])!=asset['document_revision']):
            errors.append({'block_id':asset['block_id'],'error':'Ảnh bảng không khớp bản gốc/phiên bản tài liệu'})
    if expected_images-{asset['block_id'] for asset in assets}:
        errors.append({'source':'pmc','error':'Chưa lưu đủ ảnh của các bảng không có ô văn bản'})
    report={'checked_at':now_utc(),'documents':dict(counts),'document_languages':dict(languages),
            'chunks':chunk_total,'chunks_by_source':dict(chunk_counts),'max_tokens':max_tokens,
            'block_kinds':dict(block_kinds),'chunk_flags':dict(flags),'byt_chapters':len(chapters),
            'unique_document_ids':len(ids),'unique_dois':len(dois),'unique_raw_files':len(source_hashes),
            'source_locators_checked':locator_total,
            'table_images_saved':len(assets),'table_images_not_ocr':len(assets),'review_blocks':len(reviews),
            'errors':errors,'passed':not errors,'topics':pmc.get('topics',[]),
            'validation_scope':'Cấu trúc, toàn vẹn dữ liệu và bao phủ văn bản; chưa được chuyên gia đánh giá nội dung.'}
    if not errors:
        temporary.replace(OUTPUT/'chunks.jsonl')
    save_json(target/'checks.json',report)
    if inventory:
        with (target/'documents.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(inventory[0]))
            writer.writeheader();writer.writerows(inventory)
    if reviews:
        with (target/'review_items.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(reviews[0]))
            writer.writeheader();writer.writerows(reviews)
    print(json.dumps({key:value for key,value in report.items() if key!='topics'},ensure_ascii=False,indent=2))
    return 0 if not errors else 1


if __name__=='__main__':
    raise SystemExit(main())
