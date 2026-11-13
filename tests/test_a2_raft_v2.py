import json
import shutil
from pathlib import Path

import pytest

from scripts.build_a2_raft_pilot import fingerprint,read_rows,sha,write_json,write_rows
from scripts.build_a2_raft_v2 import OUTPUT,PREVIOUS
from scripts.check_a2_raft_v2 import check
from scripts.train_a2 import load_data


@pytest.fixture
def copy_pilot(tmp_path):
    manifest=json.loads((OUTPUT/'manifest.json').read_text(encoding='utf-8'))
    for name in [*manifest['files_sha256'],'manifest.json']:
        shutil.copyfile(OUTPUT/name,tmp_path/name)
    return tmp_path


def refresh(directory,*names):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    for name in names:manifest['files_sha256'][name]=sha(directory/name)
    write_json(directory/'manifest.json',manifest)


def test_real_v2_loads_without_changing_the_validation_benchmark():
    packet=load_data(OUTPUT)
    assert len(packet['train'])==880 and len(packet['validation'])==72
    assert packet['info']['dataset_version'].startswith('a2-raft-pilot-v2-')
    assert (OUTPUT/'val.jsonl').read_bytes()==(PREVIOUS/'val.jsonl').read_bytes()
    assert (OUTPUT/'val.metadata.jsonl').read_bytes()==(PREVIOUS/'val.metadata.jsonl').read_bytes()
    assert len({r['metadata']['question_id'] for r in packet['train']})==120


def test_two_answerable_parts_cannot_silently_become_one_claim(copy_pilot):
    rows=read_rows(copy_pilot/'train.jsonl');metas=read_rows(copy_pilot/'train.metadata.jsonl')
    i=next(i for i,m in enumerate(metas) if m['mode']=='clean' and m['question_fact_indices']==[0,1])
    gold=json.loads(rows[i]['messages'][2]['content']);gold['claims'].pop()
    rows[i]['messages'][2]['content']=json.dumps(gold,ensure_ascii=False)
    metas[i]['messages_sha256']=fingerprint(rows[i]['messages'])
    write_rows(copy_pilot/'train.jsonl',rows);write_rows(copy_pilot/'train.metadata.jsonl',metas)
    refresh(copy_pilot,'train.jsonl','train.metadata.jsonl')
    with pytest.raises(ValueError,match='answer coverage'):check(copy_pilot,write_reports=False)


def test_partial_answer_must_keep_the_declared_side(copy_pilot):
    metas=read_rows(copy_pilot/'train.metadata.jsonl')
    meta=next(m for m in metas if m['mode']=='partial_answer_first')
    meta['supported_fact_indices']=[1]
    write_rows(copy_pilot/'train.metadata.jsonl',metas);refresh(copy_pilot,'train.metadata.jsonl')
    with pytest.raises(ValueError,match='answer coverage'):check(copy_pilot,write_reports=False)


def test_training_source_cannot_be_relabelled_as_validation(copy_pilot):
    seeds=json.loads((copy_pilot/'seed_review.json').read_text(encoding='utf-8'))
    next(s for s in seeds if s['split']=='train')['split']='val'
    write_json(copy_pilot/'seed_review.json',seeds);refresh(copy_pilot,'seed_review.json')
    with pytest.raises(ValueError,match='parent split'):check(copy_pilot,write_reports=False)


def test_validation_freeze_survives_manifest_checksum_refresh(copy_pilot):
    path=copy_pilot/'val.jsonl';path.write_bytes(path.read_bytes()+b'\n')
    refresh(copy_pilot,'val.jsonl')
    with pytest.raises(ValueError,match='byte-identical'):check(copy_pilot,write_reports=False)


def test_evaluation_rejects_unfrozen_weights_before_any_generation(tmp_path):
    import torch
    from types import SimpleNamespace
    from scripts.train_a2 import evaluate
    model=torch.nn.Linear(2,2);model.config=SimpleNamespace(use_cache=False)
    with pytest.raises(ValueError,match='frozen parameters'):
        evaluate(model,None,[],tmp_path/'evaluation')
