"""Local query translation only; translated text is never answer evidence."""
from collections import OrderedDict
import json
import re
from pathlib import Path
import threading


def source_language(text):
    # Prefer explicit Vietnamese characters/words, otherwise an ASCII query is English.
    if re.search(r"[ăâđêôơưĂÂĐÊÔƠƯ\u1ea0-\u1ef9]", text):
        return 'vi'
    if re.search(r"\b(what|which|how|why|when|where|does|without|definition|explain|difference|between)\b", text, re.I):
        return 'en'
    if re.search(r'\b(rang|nhai|khong|duoc|nguyen|nhan)\b',text,re.I):
        return 'vi'
    return 'en'


class NllbTranslator:
    def __init__(self, directory, threads=6):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from src.rag.local_embeddings import file_sha256
        torch.set_num_threads(threads)
        directory = Path(directory)
        manifest = json.loads((directory/'download_manifest.json').read_text(encoding='utf-8'))
        if manifest['repo'] != 'facebook/nllb-200-distilled-600M':
            raise ValueError('Sai model dịch truy vấn')
        for name, sha in manifest['sha256'].items():
            if file_sha256(directory/name) != sha:
                raise ValueError('Sai checksum model dịch: '+name)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, src_lang='vie_Latn')
        model = AutoModelForSeq2SeqLM.from_pretrained(directory, local_files_only=True, torch_dtype=torch.float32).eval()
        self.model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).eval()
        self.lock = threading.RLock()
        self.identity = {'repo':manifest['repo'], 'revision':manifest['revision'],
                         'device':'cpu', 'quantization':'dynamic_int8_linear', 'purpose':'search_query_translation_only'}

    def translate(self, query, source, target):
        import torch
        codes = {'vi':'vie_Latn', 'en':'eng_Latn'}
        with self.lock:
            self.tokenizer.src_lang = codes[source]
            inputs = self.tokenizer(query, return_tensors='pt', truncation=False)
            if inputs['input_ids'].shape[1] > 256:
                raise ValueError('Truy vấn quá dài để dịch')
            with torch.inference_mode():
                output = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(codes[target]),
                    max_new_tokens=128, do_sample=False, num_beams=2)
            if output.shape[1] >= 128 and output[0,-1].item() != self.tokenizer.eos_token_id:
                raise ValueError('Bản dịch truy vấn chưa kết thúc')
            return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()


class BilingualQueries:
    def __init__(self, translator, max_cache=128, terminology=None):
        self.translator = translator
        self.cache = OrderedDict()
        self.max_cache = max_cache
        path = Path(__file__).resolve().parents[2]/'config/dental_search_terms.json'
        self.terms = terminology if terminology is not None else json.loads(path.read_text(encoding='utf-8'))['terms']

    def protect_terms(self, query, source, target):
        aliases = {alias.casefold():term for term in self.terms for alias in term[source]}
        if not aliases:
            return query, []
        pattern = re.compile(r'(?<!\w)('+'|'.join(re.escape(a) for a in sorted(aliases,key=len,reverse=True))+r')(?!\w)',re.I)
        matched = []
        def replace(match):
            term = aliases[match.group().casefold()]
            if term not in matched:
                matched.append(term)
            return term[target][0]
        return pattern.sub(replace, query), matched

    def keywords(self, query, source, target, matched):
        """A conservative lexical search expansion, not an asserted full translation."""
        words = [term[target][0] for term in matched]
        intents = [(r'là gì|định nghĩa|what is|definition', {'vi':'định nghĩa','en':'definition'}),
                   (r'nguyên nhân|vì sao|caus|why', {'vi':'nguyên nhân','en':'causes'}),
                   (r'điều trị|chữa|treatment', {'vi':'điều trị','en':'treatment'}),
                   (r'triệu chứng|dấu hiệu|symptom', {'vi':'triệu chứng','en':'symptoms'}),
                   (r'phòng|ngăn ngừa|prevent', {'vi':'phòng ngừa','en':'prevention'}),
                   (r'khác|so sánh|difference|compare', {'vi':'so sánh','en':'comparison'})]
        words += [labels[target] for pattern,labels in intents if re.search(pattern,query,re.I)]
        words += re.findall(r'\d+(?:[.,]\d+)?',query)
        words += re.findall(r'\b[A-Z][A-Z0-9-]{1,}\b',query)
        if re.search(r'không|chưa|without|\bnot\b|\bno\b',query,re.I):
            words.append('phủ định' if target=='vi' else 'without / not')
        return ' '.join(words)

    def __call__(self, query):
        query = query.strip()
        if query in self.cache:
            self.cache.move_to_end(query)
            return self.cache[query]
        language = source_language(query)
        target = 'en' if language == 'vi' else 'vi'
        original = {'text':query, 'language':language}
        protected, matched = self.protect_terms(query, language, target)
        try:
            translated = self.translator.translate(protected, language, target)
            if not translated or len(translated)>600:
                raise ValueError('Bản dịch truy vấn không hợp lệ')
            if sorted(re.findall(r'\d+(?:[.,]\d+)?',query)) != sorted(re.findall(r'\d+(?:[.,]\d+)?',translated)):
                raise ValueError('Bản dịch truy vấn thay đổi số liệu')
            compact = re.sub(r'[\W_]+','',translated).casefold()
            if any(re.sub(r'[\W_]+','',code).casefold() not in compact
                   for code in re.findall(r'\b[A-Z][A-Z0-9-]{1,}\b',query)):
                raise ValueError('Bản dịch truy vấn làm mất mã viết tắt')
            if any(not any(re.search(r'(?<!\w)'+re.escape(alias)+(r's?' if target=='en' else '')+r'(?!\w)',translated,re.I)
                    for alias in term[target]) for term in matched):
                raise ValueError('Bản dịch truy vấn làm mất thuật ngữ đã được bảo vệ')
            if language == 'vi' and re.search(r'không\s+(?:dùng|sử dụng|có)',query,re.I):
                if not re.search(r'without|not\s+(?:using|use|have)|no\s+use',translated,re.I):
                    raise ValueError('Bản dịch chưa giữ rõ điều kiện phủ định')
            result = {'queries':[original, {'text':translated,'language':target}],
                      'translator':self.translator.identity, 'protected_input':protected,
                      'protected_terms':[{'source':term[language][0],'target':term[target][0]} for term in matched], 'error':None}
            self.cache[query] = result
            while len(self.cache)>self.max_cache:
                self.cache.popitem(last=False)
            return result
        except (ValueError,RuntimeError,OSError) as exc:
            queries = [original]
            if matched:
                queries.append({'text':self.keywords(query,language,target,matched),
                                'language':target,'kind':'terminology_keywords'})
            result = {'queries':queries, 'translator':self.translator.identity,
                      'error':type(exc).__name__+': '+str(exc)}
            if matched:
                self.cache[query] = result
                while len(self.cache)>self.max_cache:
                    self.cache.popitem(last=False)
            return result
