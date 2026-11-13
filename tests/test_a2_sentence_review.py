import pytest
from scripts.a2_sentence_review_policy import allowed_keep, additional_removal
from scripts.finalize_a2_sentence_v2 import finalize


@pytest.mark.parametrize('family,kept',[
    ('quick_master',[1]), ('articular_disc',[1]), ('passive_movement',[1]),
    ('central_relation',[0]), ('mckinney',[0]), ('epidemiology',[0]),
])
def test_repeated_or_partial_support_cannot_be_exported_as_absent(family,kept):
    assert not allowed_keep(family,kept)


def test_learning_examples_are_removed_along_with_their_fact():
    assert additional_removal('learning',[1],'This theory suggests , for example, rewarding a child.')
    assert additional_removal('learning',[0],'For example, if a child is present when a sibling receives a treat, they imitate.')
    assert not additional_removal('learning',[0,1],'This theory suggests , for example, rewarding a child.')


def test_modified_candidates_cannot_reuse_the_source_review(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'candidates.jsonl').write_text('{}\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Candidate bytes changed'):
        finalize(source,tmp_path/'release')
    assert not (tmp_path/'release').exists()
