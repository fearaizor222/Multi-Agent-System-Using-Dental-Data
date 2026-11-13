import hashlib
import json
from pathlib import Path

import pytest

from src.data import bulk_public as bulk
from src.data.sources.markup import parse_table


def test_candidate_filter_excludes_animal_reviews_and_retracted_records():
    item={"pmcid":"PMC123","isOpenAccess":"Y","language":"eng","title":"Clinical dental review",
          "pubTypeList":{"pubType":["Review"]}}
    assert bulk.candidate_reason(item) is None
    assert bulk.candidate_reason({**item,"title":"A Systematic Review of Animal Studies"})
    assert bulk.candidate_reason({**item,"isRetracted":"Y"})
    assert bulk.candidate_reason({**item,"title":"An in vitro study of dental tissue"})
    assert bulk.candidate_reason({**item,"pubTypeList":{"pubType":["Published Erratum"]}})
    # Canine có thể là răng nanh của người, không dùng từ này để loại bài động vật.
    assert bulk.candidate_reason({**item,"title":"Orthodontic canine retraction review"}) is None


def test_empty_html_spans_default_to_one_cell():
    table=parse_table('<table><tr><td rowspan="" colspan="">A</td><td>B</td></tr></table>')
    assert [(cell.column,cell.rowspan,cell.colspan) for cell in table.cells]==[(0,1,1),(1,1,1)]


def test_windows_newline_repair_requires_original_checksum(tmp_path):
    path=tmp_path/'article.html'
    original='<html>\r\n<p>Nội dung</p>\n</html>'
    path.write_bytes(original.replace('\n','\r\n').encode('utf-8'))
    entry={'html_path':str(path),'sha256':bulk.digest(original.encode('utf-8'))}
    assert bulk.cached_html(entry)==original
    assert path.read_bytes()==original.encode('utf-8')
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='cache'):
        bulk.cached_html(entry)


def test_cloud_chooses_published_version_verifies_md5_and_resumes(tmp_path,monkeypatch):
    monkeypatch.setattr(bulk,"RAW",tmp_path)
    xml=b'<article><body><p>Original XML</p></body></article>'
    checksum=hashlib.md5(xml).hexdigest()
    calls=[]

    class Response:
        def __init__(self,content=None,data=None):
            self.content=content
            self.data=data
        def json(self):
            return self.data

    class Client:
        def get(self,url,**kwargs):
            calls.append(url)
            if url==bulk.S3_BASE:
                return Response(b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><CommonPrefixes><Prefix>PMC123.1/</Prefix></CommonPrefixes><CommonPrefixes><Prefix>PMC123.2/</Prefix></CommonPrefixes></ListBucketResult>')
            if url.endswith('.json'):
                version=2 if 'PMC123.2' in url else 1
                return Response(data={"pmcid":"PMC123","version":version,"is_pmc_openaccess":True,
                    "is_retracted":False,"is_historical_ocr":False,"is_manuscript":version==2,
                    "xml_url":f's3://pmc-oa-opendata/PMC123.{version}/PMC123.{version}.xml?md5={checksum}'})
            assert 'PMC123.1/' in url
            return Response(xml)

    path,receipt=bulk.cloud_article(Client(),'PMC123')
    assert path.read_bytes()==xml and receipt['metadata']['version']==1
    before=len(calls)
    assert bulk.cloud_article(Client(),'PMC123')[0]==path
    assert len(calls)==before
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='cache'):
        bulk.cloud_article(Client(),'PMC123')
