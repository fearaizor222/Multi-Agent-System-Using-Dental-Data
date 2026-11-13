"""Kiểm tra sampler, trọng số loss và bảo vệ split khi sửa cách train."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts.check_a1_environment import completion_only_loss
from scripts.prepare_a1_dataset import build_records
from src.training.a1_runtime import ROOT
from src.training.a1_strategy import assistant_weights, plan_epoch, training_score


def train_rows():
    source=json.loads((ROOT/"data/curated/a1_supervisor_v3_1/scenarios.json").read_text(encoding="utf-8"))
    return [r for r in build_records(source) if r["split"]=="train"]


def config():
    return json.loads((ROOT/"config/a1_training_v4.json").read_text(encoding="utf-8"))


def test_balanced_plan_covers_all_train_and_has_exact_reproducible_quotas():
    rows=train_rows()
    order,stats=plan_epoch(rows,77,config()["sampling"])
    assert set(order)==set(range(len(rows)))
    assert stats["samples"]==900 and stats["unique_train_samples"]==618
    assert stats["decision_counts"]=={"route":360,"clarify":225,"handoff":180,"out_of_scope":68,"respond":67}
    assert plan_epoch(rows,77,config()["sampling"])[0]==order
    assert plan_epoch(rows,78,config()["sampling"])[0]!=order
    assert all(rows[i]["split"]=="train" for i in order)


def test_sampler_rejects_holdout_and_quota_that_discards_training_examples():
    rows=train_rows()
    rows[0]["split"]="validation"
    with pytest.raises(ValueError,match="chỉ nhận mẫu train"):
        plan_epoch(rows,1,config()["sampling"])
    sampling=deepcopy(config()["sampling"])
    sampling["samples_per_epoch"]=618
    with pytest.raises(ValueError,match="Quota"):
        plan_epoch(train_rows(),1,sampling)


def test_weighted_sparse_loss_matches_full_logits_loss_and_gradients():
    import torch
    torch.manual_seed(9)
    full=torch.randn(1,10,17,requires_grad=True)
    sparse=full.detach().clone().requires_grad_()
    tokens=torch.randint(0,17,(1,10))
    labels=tokens.clone();labels[:,:6]=-100
    weights=torch.tensor([[12.,3.,1.,1.]])
    batch={"input_ids":tokens,"attention_mask":torch.ones_like(tokens),"labels":labels}
    losses=torch.nn.functional.cross_entropy(full[:,:-1].reshape(-1,17),labels[:,1:].reshape(-1),ignore_index=-100,reduction="none")
    reference=(losses[5:]*weights.reshape(-1)).sum()/weights.sum()
    class Fake:
        def __call__(self,**kwargs):
            return SimpleNamespace(logits=sparse[:,kwargs["logits_to_keep"],:])
    selected=completion_only_loss(Fake(),batch,6,weights)
    assert torch.allclose(reference,selected)
    reference.backward();selected.backward()
    assert torch.allclose(full.grad,sparse.grad)
    with pytest.raises(ValueError,match="Trọng số"):
        completion_only_loss(Fake(),batch,6,torch.tensor([[0.,1.,1.,1.]]))


def test_weights_align_to_real_answer_tokens_and_not_eos_or_free_text():
    raw='{"decision":"clarify","targets":[],"service_topics":[],"image_ids":[],"resolved_request":null,"message":"ask"}'
    class Characters:
        def __call__(self,text,**kwargs):
            return {"input_ids":[ord(c) for c in text],"offset_mapping":[(i,i+1) for i in range(len(text))]}
    ids=[777]+[ord(c) for c in raw]+[999]
    weights=assistant_weights(Characters(),[{"content":raw}],ids,1,config()["token_weighting"])
    assert weights[raw.index('clarify')]==12
    assert weights[raw.index('targets')]==3
    assert weights[raw.index('ask')]==1 and weights[-1]==1
    with pytest.raises(ValueError,match="không khớp"):
        assistant_weights(Characters(),[{"content":raw}],[777,888],1,config()["token_weighting"])


def test_critical_selection_does_not_reward_zero_clarify_recall_over_balanced_candidate():
    a={"rates":{"schema_valid":1.,"context_valid":1.,"control_exact":.68},"decision_macro_f1":.75,
       "per_decision":{"clarify":{"recall":0.},"handoff":{"recall":.75}}}
    b=deepcopy(a)
    b["per_decision"]["clarify"]["recall"]=.67
    b["per_decision"]["handoff"]["recall"]=1.
    b["rates"]["control_exact"]=.65
    assert training_score(b,"critical_recall")>training_score(a,"critical_recall")


def test_v3_changes_only_train_and_audits_each_wording_revision():
    before={r["id"]:r for r in build_records(json.loads((ROOT/"data/curated/a1_supervisor_v2/scenarios.json").read_text(encoding="utf-8")))}
    after={r["id"]:r for r in build_records(json.loads((ROOT/"data/curated/a1_supervisor_v3_1/scenarios.json").read_text(encoding="utf-8")))}
    revisions=json.loads((ROOT/"data/curated/a1_supervisor_v3_1/train_revisions.json").read_text(encoding="utf-8"))
    changed={r["id"] for r in revisions}
    assert len(changed)==151
    for name,old in before.items():
        new=after[name]
        if name in changed:
            assert old["split"]==new["split"]=="train"
            assert old["input"]==new["input"]
            assert new["target"]["resolved_request"]==new["input"]["message"]
            assert {k:v for k,v in old["target"].items() if k!="resolved_request"}=={k:v for k,v in new["target"].items() if k!="resolved_request"}
        else:
            assert old==new


def test_injection_cases_keep_sanitized_request_instead_of_copying_attack():
    rows=train_rows()
    injections=[r for r in rows if r["group_id"].startswith("a1_inject_")]
    assert len(injections)==9
    for r in injections:
        assert r["target"]["resolved_request"]!=r["input"]["message"]
        assert all(term not in r["target"]["resolved_request"] for term in ("SYSTEM","A3","A4","bịa","giả vờ","thành công"))
