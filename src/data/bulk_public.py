"""Thu thập theo lô, có cache, sổ lỗi và khả năng chạy tiếp."""
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import requests

from src.data.sources.msd_http import START_URL, USER_AGENT, CrawlStopped, get_html, load_robots, normalize_url
from src.data.public_schema import PublicDocument, digest, make_document, now_utc, write_documents
from src.data.pmc_topics import PMC_TOPICS
from src.data.sources.msd import parse_msd
from src.data.sources.pmc import parse_pmc

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT/"data/raw/public_v1"
OUTPUT = ROOT/"data/processed/public_v1/corpus"
S3_BASE = "https://pmc-oa-opendata.s3.amazonaws.com/"
SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def load_documents(path):
    if not path.exists():
        return []
    return [PublicDocument.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cached_html(entry):
    path=Path(entry["html_path"])
    data=path.read_bytes()
    if digest(data)!=entry["sha256"]:
        # Khôi phục việc Windows tự đổi LF thành CRLF của lần lưu đầu; chỉ sửa khi khớp hash gốc.
        restored=data.decode("utf-8").replace("\r\n","\n").encode("utf-8")
        if digest(restored)!=entry["sha256"]:
            raise ValueError("HTML cache đã thay đổi")
        path.write_bytes(restored)
        data=restored
    return data.decode("utf-8")


class HttpClient:
    def __init__(self, delay=0.4):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent":"C20DentalResearch/0.3", "Accept-Encoding":"gzip, deflate"})
        self.delay, self.last_request = delay, 0

    def get(self, url, **kwargs):
        for attempt in range(3):
            time.sleep(max(0, self.delay-(time.monotonic()-self.last_request)))
            self.last_request = time.monotonic()
            response = self.session.get(url, timeout=(15,60), **kwargs)
            if response.status_code in {401,403,429}:
                raise CrawlStopped(f"HTTP {response.status_code}; dừng dịch vụ, giữ dữ liệu đã tải")
            if response.status_code not in {500,502,503,504}:
                response.raise_for_status()
                return response
            if attempt < 2:
                time.sleep(2**(attempt+1))
        response.raise_for_status()


def collect_msd(output=OUTPUT):
    output = Path(output)
    state_path = RAW/"msd/crawl_state.json"
    state = load_json(state_path, {"pages":{}})
    report = {"source":"msd", "started_at":now_utc(), "errors":[], "stopped_reason":None}
    documents, seen, queue = {}, set(), [normalize_url(START_URL)]
    with requests.Session() as session:
        session.headers.update({"User-Agent":USER_AGENT,"Accept-Language":"vi"})
        robots = load_robots(session)
        delay = max(2, robots.crawl_delay(USER_AGENT) or 0)
        try:
            while queue:
                url = queue.pop(0)
                if url in seen:
                    continue
                seen.add(url)
                try:
                    cached = state["pages"].get(url)
                    if cached and Path(cached["html_path"]).exists():
                        path, stamp = Path(cached["html_path"]), cached["retrieved_at"]
                        html = cached_html(cached)
                    else:
                        html, final_url = get_html(session, robots, url, delay)
                        path = RAW/"msd"/(digest(html.encode("utf-8"))+".html")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(html.encode("utf-8"))
                        stamp = now_utc()
                        state["pages"][url] = {"html_path":str(path),"retrieved_at":stamp,
                                               "sha256":digest(html.encode("utf-8")),"final_url":final_url}
                        save_json(state_path,state)
                    # Chỉ theo liên kết trong mục lục/nhóm nha khoa. Bài bệnh không làm mở rộng phạm vi.
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html,"html.parser")
                    if soup.select_one("#topicHeaderTitle") and soup.select_one('[data-testid="topic-main-content"]'):
                        doc = parse_msd(path, RAW/"msd", retrieved_at=stamp)
                        documents[doc.document_id] = doc
                        write_documents(output/"msd_documents.jsonl", list(documents.values()))
                        print(f"MSD {len(documents)}: {doc.title}",flush=True)
                    else:
                        for link in soup.select("a[href]"):
                            child = normalize_url(link.get("href",""),url)
                            if child and child not in seen and child not in queue:
                                queue.append(child)
                except CrawlStopped:
                    raise
                except (ValueError,OSError,requests.RequestException) as exc:
                    report["errors"].append({"url":url,"error":str(exc)})
                    print(f"MSD cần xem: {url}: {exc}",flush=True)
        except CrawlStopped as exc:
            report["stopped_reason"] = str(exc)
        finally:
            report.update(finished_at=now_utc(),visited_pages=len(seen),documents=len(documents),
                          discovered_pages=len(seen)+len(queue),pending_urls=queue,
                          catalog_urls=[url for url in seen if url not in {doc.source_url for doc in documents.values()}])
            save_json(output/"msd_collection.json",report)
    return report


ANIMAL_OR_LAB = re.compile(r"\b(animals?|rats?|mice|murine|dogs?|rabbits?|porcine|bovine|zebrafish|in[ -]vitro|ex[ -]vivo|preclinical)\b",re.I)
UNSUITABLE = re.compile(r"^(?:correction|corrigendum|erratum|retraction)\b|\bstudy protocol\b|\bcase report\b",re.I)
NONCLINICAL = re.compile(r"\b(veterinar\w*|bibliometric\w*|scientometric\w*|chatgpt)\b|large language models?",re.I)
TITLE_PATTERNS = {
    "caries":r"caries|cariogenic|tooth decay",
    "endodontics":r"pulpitis|apical|endodont|root canal|pulpotom|pulp capping",
    "gingivitis":r"gingiv|plaque|biofilm",
    "periodontitis":r"periodont|gingiv",
    "odontogenic_infection":r"odontogenic|dental abscess|periapical abscess|dental infection|endodontic infection|maxillofacial infection|Ludwig|deep neck",
    "third_molar":r"third molar|wisdom|pericoron",
    "oral_mucosa":r"lichen planus|leukoplakia|aphthous|oral candid|oral mucos|oral ulcer",
    "xerostomia":r"xerostom|hyposaliv|dry mouth|salivary",
    "halitosis":r"halitosis|oral malodo|bad breath",
    "dental_trauma":r"dental traum|tooth avuls|teeth avuls|traumatic dental|tooth luxation|teeth luxation|dental injuries",
    "tmd":r"temporomandibular|orofacial pain",
    "tooth_wear":r"tooth wear|dental eros|dentin.*sensitiv|dentine.*sensitiv|tooth eros|erosive tooth|dentinal.*sensitiv",
}


def title_topic_reason(title,topic_id):
    if not re.search(TITLE_PATTERNS[topic_id],title,re.I):
        return "tiêu đề không khớp chủ đề đang thu thập"
    return None


def candidate_reason(item):
    types = " ".join(item.get("pubTypeList",{}).get("pubType",[])).lower()
    if not re.fullmatch(r"PMC\d+",item.get("pmcid") or "") or item.get("isOpenAccess") != "Y":
        return "không có PMCID/toàn văn Open Access"
    if item.get("isRetracted")=="Y" or any(term in types for term in ("retraction","erratum","correction","preprint","protocol","case report")):
        return "bài rút lại/đính chính/preprint/protocol/ca đơn lẻ"
    if ANIMAL_OR_LAB.search(item.get("title","")):
        return "tiêu đề là nghiên cứu động vật hoặc trong phòng thí nghiệm"
    if UNSUITABLE.search(item.get("title","")):
        return "loại bài không chọn cho kho đối chiếu ban đầu"
    if NONCLINICAL.search(item.get("title","")):
        return "thú y hoặc nghiên cứu thư mục/mô hình ngôn ngữ ngoài phạm vi đối chiếu lâm sàng"
    if item.get("language") not in {"eng","vie"}:
        return "ngôn ngữ ngoài phạm vi Việt/Anh"
    return None


def candidate_priority(item):
    text = (item.get("title","")+" "+" ".join(item.get("pubTypeList",{}).get("pubType",[]))).lower()
    score = 3 if "guideline" in text or "consensus" in text else 2 if "systematic review" in text or "meta-analysis" in text else 1 if "review" in text else 0
    return (score,int(item.get("pubYear") or 0))


def search_topic(client,topic,end_date):
    query = f'({topic["query"]}) AND IN_PMC:Y AND OPEN_ACCESS:Y AND FIRST_PDATE:[2015-01-01 TO {end_date}]'
    cache = RAW/"pmc/search"/(topic["id"]+"_"+digest(query.encode("utf-8"))[:16]+".json")
    if cache.exists():
        data = load_json(cache,{})
    else:
        response = client.get(SEARCH_URL,params={"query":query,"format":"json","pageSize":100,"resultType":"core"})
        data = response.json()
        data["collected_at"] = now_utc()
        save_json(cache,data)
    results = data.get("resultList",{}).get("result",[])
    # Lấy thêm trang kết quả riêng cho review để không chỉ chọn bài mới nhất.
    review_query = query + ' AND PUB_TYPE:Review'
    review_cache = RAW/"pmc/search"/(topic["id"]+"_reviews_"+digest(review_query.encode("utf-8"))[:16]+".json")
    reviews = load_json(review_cache,None)
    if reviews is None:
        reviews = client.get(SEARCH_URL,params={"query":review_query,"format":"json","pageSize":100,"resultType":"core"}).json()
        reviews["collected_at"] = now_utc()
        save_json(review_cache,reviews)
    results += reviews.get("resultList",{}).get("result",[])
    results = list({item.get("pmcid") or item.get("id"):item for item in results}.values())
    return sorted(results,key=candidate_priority,reverse=True),query,data.get("hitCount")


def cloud_article(client,pmcid):
    """PMC Cloud Service; đọc phiên bản và giấy phép thay vì đoán URL version 1."""
    receipt_path = RAW/"pmc/cloud"/(pmcid+".receipt.json")
    if receipt_path.exists():
        receipt = load_json(receipt_path,{})
        path = Path(receipt["xml_path"])
        if path.exists() and digest(path.read_bytes()) == receipt["sha256"]:
            return path,receipt
        raise ValueError("XML cache không khớp receipt")
    response = client.get(S3_BASE,params={"list-type":2,"prefix":pmcid+".","delimiter":"/"})
    root = ET.fromstring(response.content)
    prefixes = [node.text for node in root.findall("{*}CommonPrefixes/{*}Prefix")]
    versions = []
    for prefix in prefixes:
        filename = prefix.rstrip("/")
        metadata_url = S3_BASE+prefix+filename+".json"
        metadata = client.get(metadata_url).json()
        save_json(RAW/"pmc/cloud"/(filename+".json"),metadata)
        if metadata.get("is_pmc_openaccess") and not metadata.get("is_retracted") and not metadata.get("is_historical_ocr"):
            versions.append((metadata,metadata_url))
    if not versions:
        raise ValueError("Không có phiên bản Open Access phù hợp trên PMC Cloud")
    # Ưu tiên bản xuất bản; version chỉ dùng phá hòa trong cùng loại.
    metadata,metadata_url = sorted(versions,key=lambda pair:(bool(pair[0].get("is_manuscript")),-int(pair[0]["version"])))[0]
    parsed = urlsplit(metadata["xml_url"])
    if parsed.scheme != "s3" or parsed.netloc != "pmc-oa-opendata":
        raise ValueError("URL XML không thuộc PMC Cloud Service")
    url = S3_BASE+parsed.path.lstrip("/")+ ("?"+parsed.query if parsed.query else "")
    response = client.get(url)
    md5 = parse_qs(parsed.query).get("md5",[None])[0]
    if md5 and hashlib.md5(response.content).hexdigest()!=md5:
        raise ValueError("XML tải về không khớp MD5 của PMC")
    path = RAW/"pmc/cloud"/(pmcid+"."+str(metadata["version"])+".xml")
    path.write_bytes(response.content)
    receipt = {"xml_path":str(path),"xml_url":url,"metadata_url":metadata_url,"metadata":metadata,
               "sha256":digest(response.content),"retrieved_at":now_utc()}
    save_json(receipt_path,receipt)
    return path,receipt


def collect_pmc(target=200,output=OUTPUT,end_date="2026-10-15"):
    if not 200 <= target <= 500:
        raise ValueError("Số bài PMC cần trong khoảng 200–500")
    output = Path(output)
    retained,removed = {},[]
    for doc in load_documents(output/"pmc_documents.jsonl"):
        item = {"pmcid":doc.document_id,"isOpenAccess":"Y","language":{"en":"eng","vi":"vie"}.get(doc.language),
                "title":doc.title,"pubTypeList":{"pubType":doc.source_metadata.get("pubmed_publication_types",[])}}
        reason = candidate_reason(item) or title_topic_reason(doc.title,doc.source_metadata["collection_topic"])
        if reason:
            removed.append({"pmcid":doc.document_id,"title":doc.title,"reason":reason})
        else:
            retained[doc.document_id] = doc
    docs = retained
    dois = {str(doc.source_metadata.get("doi") or "").lower() for doc in docs.values()}
    dois.discard("")
    topics = PMC_TOPICS
    report = {"source":"pmc","started_at":now_utc(),"target":target,"topics":[],"rejected":removed,"errors":[],"stopped_reason":None,
              "search_service":"Europe PMC metadata API","full_text_service":"NIH/NLM PMC Cloud Service"}
    client = HttpClient()
    try:
        for index,topic in enumerate(topics):
            quota = target//len(topics) + (index < target%len(topics))
            selected = [doc for doc in docs.values() if doc.source_metadata.get("collection_topic")==topic["id"]]
            status = {"id":topic["id"],"name":topic["name"],"quota":quota,"accepted":len(selected),"candidates":0}
            report["topics"].append(status)
            if len(selected)>=quota:
                continue
            items,query,hits = search_topic(client,topic,end_date)
            status.update(query=query,search_hits=hits,candidates=len(items))
            for item in items:
                if status["accepted"]>=quota:
                    break
                pmcid = item.get("pmcid")
                reason = candidate_reason(item) or title_topic_reason(item.get("title",""),topic["id"])
                if reason:
                    report["rejected"].append({"pmcid":pmcid,"title":item.get("title"),"topic":topic["id"],"reason":reason})
                    continue
                doi = str(item.get("doi") or "").lower()
                if pmcid in docs or (doi and doi in dois):
                    continue
                try:
                    path,receipt = cloud_article(client,pmcid)
                    doc = parse_pmc(path,RAW/"pmc",receipt["xml_url"],receipt["retrieved_at"])
                    if doc.document_id != pmcid:
                        raise ValueError("PMCID trong XML không khớp bài đã chọn")
                    if doc.language not in {"en","vi"}:
                        raise ValueError("Ngôn ngữ XML ngoài phạm vi Việt/Anh")
                    if doc.source_metadata.get("article_type") in {"correction","retraction","retraction-notice","expression-of-concern","editorial","letter"}:
                        raise ValueError("Loại bài XML không phù hợp với kho đối chiếu ban đầu")
                    values = doc.model_dump(exclude={"revision_id"})
                    values["source_metadata"].update(collection_topic=topic["id"],collection_query=query,
                        search_service="Europe PMC",pubmed_publication_types=item.get("pubTypeList",{}).get("pubType",[]),
                        pmc_cloud_version=receipt["metadata"]["version"],license_code=receipt["metadata"].get("license_code"),
                        pmc_cloud_metadata_url=receipt["metadata_url"],selection_reviewed_by_expert=False)
                    doc = make_document(**values)
                    docs[pmcid] = doc
                    if doi:
                        dois.add(doi)
                    status["accepted"]+=1
                    write_documents(output/"pmc_documents.jsonl",list(docs.values()))
                    print(f'PMC {len(docs)}/{target} [{topic["id"]}]: {doc.title}',flush=True)
                    save_json(output/"pmc_collection.json",report)
                except CrawlStopped:
                    raise
                except (ValueError,OSError,requests.RequestException,ET.ParseError) as exc:
                    report["errors"].append({"pmcid":pmcid,"topic":topic["id"],"error":str(exc)})
                    print(f"PMC cần xem {pmcid}: {exc}",flush=True)
        report["complete"] = len(docs)>=target and all(topic["accepted"]>=topic["quota"] for topic in report["topics"])
    except (CrawlStopped,ValueError,requests.RequestException) as exc:
        report["stopped_reason"] = str(exc)
        report["complete"] = False
    finally:
        client.session.close()
        report.update(documents=len(docs),finished_at=now_utc())
        save_json(output/"pmc_collection.json",report)
    return report
