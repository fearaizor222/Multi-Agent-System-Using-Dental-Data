"""Chạy lại parser từ bản gốc đã lưu, không gọi mạng."""
import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.bulk_public import OUTPUT,RAW,load_documents
from src.data.public_schema import digest,make_document,write_documents
from src.data.sources.byt import parse_byt
from src.data.sources.msd import parse_msd
from src.data.sources.pmc import parse_pmc


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=['byt','msd','pmc'])
    args=parser.parse_args()
    for source in ([args.source] if args.source else ('byt','msd','pmc')):
        path=OUTPUT/(source+'_documents.jsonl')
        records=[]
        for previous in load_documents(path):
            raw_path=Path(previous.raw.path)
            if digest(raw_path.read_bytes())!=previous.raw.sha256:
                raise ValueError(f'Bản gốc đã thay đổi: {previous.document_id}')
            if source=='byt':
                doc=parse_byt(raw_path,RAW/'byt',previous.source_metadata['selected_pdf_pages'])
                values=doc.model_dump(exclude={'revision_id'})
                values['source_metadata']['original_filename']=previous.source_metadata['original_filename']
                doc=make_document(**values)
            elif source=='msd':
                doc=parse_msd(raw_path,RAW/'msd',previous.raw.retrieved_at)
            else:
                doc=parse_pmc(raw_path,RAW/'pmc',previous.raw.acquisition_url,previous.raw.retrieved_at)
                values=doc.model_dump(exclude={'revision_id'})
                for key in ('collection_topic','collection_query','search_service','pubmed_publication_types',
                            'pmc_cloud_version','license_code','pmc_cloud_metadata_url','selection_reviewed_by_expert'):
                    if key in previous.source_metadata:
                        values['source_metadata'][key]=previous.source_metadata[key]
                doc=make_document(**values)
            records.append(doc)
        write_documents(path,records)
        print(f'{source}: xử lý lại {len(records)} tài liệu',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
