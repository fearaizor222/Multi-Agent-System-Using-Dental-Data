import copy
import json
import pytest
from scripts.a2_hard_specs import specs
from scripts.build_a2_hard_v3 import make_family,ROOT


def sample():
    s=specs()[0]
    catalog=json.loads((ROOT/'data/curated/a2_hard_v3_sources.json').read_text(encoding='utf-8'))
    return s,catalog[str(s['pool_index'])]


def test_refusal_and_partial_use_the_same_unedited_source_as_answerable_questions():
    s,source=sample();rr=make_family(s,source)
    evidence=[json.JSONDecoder().raw_decode(r['messages'][1]['content'])[0]['evidence'] for r in rr]
    assert all(e==evidence[0] for e in evidence)
    negative=next(r for r in rr if r['metadata']['kind']=='intact_same_entity_missing_aspect')
    partial=next(r for r in rr if r['metadata']['kind']=='intact_partial_support')
    assert json.loads(negative['messages'][-1]['content'])=={'status':'insufficient_evidence','claims':[]}
    assert len(json.loads(partial['messages'][-1]['content'])['claims'])==1


def test_different_aspects_cannot_share_the_same_gold_unit():
    s,source=sample();s=copy.deepcopy(s);s['facts'][1]['needle']=s['facts'][0]['needle']
    with pytest.raises(ValueError,match='facts share target'):make_family(s,source)


def test_ambiguous_evidence_is_not_silently_used():
    s,source=sample();s=copy.deepcopy(s);s['facts'][0]['needle']='.'
    with pytest.raises(ValueError,match='ambiguous/missing'):make_family(s,source)
