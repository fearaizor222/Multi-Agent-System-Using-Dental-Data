import ast
import json

import pytest

from scripts.prepare_a2_colab import notebook
from scripts.train_a2_colab import CHECKPOINT_REQUIRED,checkpoint_manifest,latest_complete_checkpoint,mixed_training_records


def test_replay_draws_are_balanced_deterministic_and_do_not_change_source_rows():
    raft=[{'kind':'raft','id':i} for i in range(880)]
    replay=[{'kind':'replay','id':i} for i in range(31)]
    result=mixed_training_records(raft,replay,220,42)
    assert result==mixed_training_records(raft,replay,220,42)
    assert len(result)==1100 and sum(r['kind']=='raft' for r in result)==880
    counts=[sum(r['kind']=='replay' and r['id']==i for r in result) for i in range(31)]
    assert min(counts)==7 and max(counts)==8
    assert len(replay)==31 and len(raft)==880


def test_resume_ignores_partial_checkpoint_and_rejects_corrupted_state(tmp_path):
    complete=tmp_path/'checkpoint-25';complete.mkdir()
    for name in CHECKPOINT_REQUIRED:(complete/name).write_bytes(b'saved state')
    checkpoint_manifest(complete,25)
    incomplete=tmp_path/'checkpoint-50';incomplete.mkdir()
    (incomplete/'adapter_model.safetensors').write_bytes(b'half written')
    assert latest_complete_checkpoint(tmp_path)==complete
    (complete/'optimizer.pt').write_bytes(b'corrupted')
    assert latest_complete_checkpoint(tmp_path) is None


def test_adapter_without_optimizer_cannot_be_marked_resumable(tmp_path):
    (tmp_path/'adapter_model.safetensors').write_bytes(b'adapter')
    with pytest.raises(ValueError,match='resume state'):checkpoint_manifest(tmp_path,25)


def test_notebook_python_cells_parse_and_do_not_use_modal():
    document=notebook()
    assert document['metadata']['accelerator']=='GPU'
    assert len({c['id'] for c in document['cells']})==len(document['cells'])
    for cell in document['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        if not source.startswith('%pip'):ast.parse(source)
        assert 'import modal' not in source
