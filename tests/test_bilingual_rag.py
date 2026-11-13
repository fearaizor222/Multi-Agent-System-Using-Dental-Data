from types import SimpleNamespace
import pytest
from src.rag.base import DentalDocument
from src.rag.bilingual_queries import BilingualQueries, source_language
from src.rag.reranked_retriever import RerankedRetriever
from src.rag.adjacent_context import expand_adjacent


class Translator:
    identity = {'repo':'test-translator'}
    def __init__(self, output='What is dental caries?'):
        self.output, self.calls = output, 0
    def translate(self,*args):
        self.calls += 1
        return self.output


def document(key, language='vi', provider='book', content='Source content'):
    return DentalDocument(id=key,title='Title',content=content,metadata={
        'warehouse':'internal','language':language,'document_id':key,'source':provider,'section_id':'s1'})


def test_query_translation_is_cached_and_preserves_original():
    translator = Translator()
    planner = BilingualQueries(translator)
    result = planner('Sâu răng là gì?')
    assert result['queries'][0]['text'] == 'Sâu răng là gì?'
    assert result['queries'][1] == {'text':'What is dental caries?','language':'en'}
    assert planner('Sâu răng là gì?') == result and translator.calls == 1


def test_translation_changed_numbers_falls_back_with_explicit_error():
    planner = BilingualQueries(Translator('Tooth 48'),terminology=[])
    result = planner('Răng 38 là gì?')
    assert len(result['queries']) == 1 and 'số liệu' in result['error']


def test_wrong_dental_translation_is_never_used_as_search_query():
    planner = BilingualQueries(Translator("What's a toothbrush?"))
    result = planner('Sâu răng là gì?')
    assert 'thuật ngữ' in result['error']
    assert result['queries'][1]['text'] == 'dental caries definition'
    assert result['queries'][1]['kind'] == 'terminology_keywords'


def test_negated_condition_uses_keywords_when_translation_changes_its_scope():
    planner = BilingualQueries(Translator('Can fluoride not prevent dental caries?'))
    result = planner('Không dùng fluor có ngăn ngừa sâu răng được không?')
    assert 'phủ định' in result['error']
    assert result['queries'][0]['text'].startswith('Không dùng')
    assert 'without / not' in result['queries'][1]['text']


def test_missing_acronyms_are_preserved_in_keyword_fallback():
    planner = BilingualQueries(Translator('What is a dental articulator?'))
    result = planner('Giá khớp ARCON và NON-ARCON khác nhau thế nào?')
    assert 'mã viết tắt' in result['error']
    assert 'ARCON' in result['queries'][1]['text'] and 'NON-ARCON' in result['queries'][1]['text']


@pytest.mark.parametrize('query,language',[('Máng nhai là gì?','vi'),('What is dental caries?','en'),
    ('Occlusal splint','en'),('sau rang la gi','vi')])
def test_query_language(query,language):
    assert source_language(query)==language


def test_explicit_new_topic_does_not_retrieve_previous_topic():
    from src.graph.assistant_runtime import select_retrieval_query
    query, strategy=select_retrieval_query('Sâu răng là gì?',
        [{'role':'user','content':'Máng nhai là gì?'}],lambda text:True)
    assert query=='Sâu răng là gì?' and strategy=='explicit_current_topic'


class Hybrid:
    def __init__(self):
        self.documents = [document('vi'), document('en','en')]
        self.calls = []
    def retrieve(self,query,**options):
        self.calls.append((query,options))
        return [self.documents[1] if query.startswith('What') else self.documents[0]]


class Reranker:
    tokenizer = staticmethod(lambda text,**kwargs:{'input_ids':text.split()})
    def __init__(self):self.calls=[]
    def rerank(self,query,docs,top_k):
        self.calls.append((query,[d.id for d in docs]))
        result=[]
        for d in docs[:top_k]:
            copy=d.model_copy(deep=True)
            # Different query logits have deliberately incomparable scales.
            copy.score=-100 if query.startswith('What') else 100
            copy.metadata['retrieval']={'rerank_logit':copy.score}
            result.append(copy)
        return result


def test_english_source_is_ranked_with_english_query_and_not_lost_to_logit_scale():
    hybrid, ranker = Hybrid(), Reranker()
    retriever = RerankedRetriever(hybrid, ranker, 30, BilingualQueries(Translator()))
    hits = retriever.retrieve('Sâu răng là gì?',top_k=2)
    assert [d.id for d in hits]==['vi','en']
    assert ('What is dental caries?',['en']) in ranker.calls
    assert all(c[1]['top_k']==30 for c in hybrid.calls)
    assert all(d.metadata['warehouse']=='internal' for d in hits)
    assert all(d.content=='Source content' for d in hits)


def test_independent_source_exclusions_apply_to_every_language_branch():
    hybrid, ranker = Hybrid(), Reranker()
    retriever = RerankedRetriever(hybrid, ranker, 30, BilingualQueries(Translator()))
    hits = retriever.retrieve_excluding('Sâu răng là gì?',document_ids={'en'})
    assert [d.id for d in hits]==['vi']
    assert all('en' not in ids for _,ids in ranker.calls)


def test_relevance_filter_can_return_empty_instead_of_forcing_top_k():
    hybrid, ranker = Hybrid(), Reranker()
    for doc in hybrid.documents:
        doc.metadata['language']='en'
    retriever=RerankedRetriever(hybrid,ranker,30,BilingualQueries(Translator()))
    retriever.min_rerank_logit=0.0
    assert retriever.retrieve('Sâu răng là gì?',top_k=3)==[]
    assert len(retriever.last_run['relevance_filter']['dropped'])==2


def test_relevance_filter_does_not_change_source_identity():
    retriever=RerankedRetriever(Hybrid(),Reranker(),30,BilingualQueries(Translator()))
    retriever.min_rerank_logit=0.0
    hits=retriever.retrieve('Sâu răng là gì?',top_k=3)
    assert [d.id for d in hits]==['vi']
    assert hits[0].metadata['warehouse']=='internal' and hits[0].content=='Source content'


def test_continuation_keeps_original_chunks_and_never_crosses_document():
    overlap='This is an exact long source paragraph which was cut in the middle and continues'
    first=document('first','en',content='Title\nEarlier content. '+overlap)
    second=document('second','en',content='Title\n'+overlap+' on the next chunk.')
    second.metadata['document_id']='first'
    expanded=expand_adjacent([first,second],[first])
    assert [d.id for d in expanded]==['first','second']
    assert expanded[1].content==second.content and 'retrieval' not in second.metadata
    second.metadata['document_id']='unrelated-book'
    assert expand_adjacent([first,second],[first])==[first]
