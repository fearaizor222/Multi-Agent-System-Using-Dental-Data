import json
from pathlib import Path

import pytest

from scripts.prepare_a1_mediphi_modal import CONFIG, load_payload, materialize
from scripts.check_a1_environment import completion_only_loss, encode_assistant_sample
from src.training.a1_model_support import load_tokenizer
from src.training.a1_runtime import ROOT
from src.training.a1_strategy import assistant_weights


def test_config_keeps_a1_comparison_settings_and_uses_phi_modules():
    c=json.loads(CONFIG.read_text(encoding='utf-8'))
    old=json.loads((ROOT/'config/a1_qwen3_17b_v4.json').read_text(encoding='utf-8'))
    differences={k for k in set(c)|set(old) if c.get(k)!=old.get(k)}
    assert differences=={'purpose','base_model','revision','model_dir','max_length','target_modules'}
    assert c['max_length']==2048 and c['target_modules']==['qkv_proj','o_proj','gate_up_proj','down_proj']


def test_source_and_splits_recreated_exactly_on_linux_style_root(tmp_path):
    packet=load_payload()
    config=materialize(packet,tmp_path)
    assert config==packet['config']
    assert packet['dataset_check']['by_split']=={'train':802,'validation':59,'test':59}


def test_real_mediphi_tokenizer_masks_and_weights_all_train_without_truncation():
    c=json.loads(CONFIG.read_text(encoding='utf-8'))
    tokenizer=load_tokenizer(c,ROOT)
    rows=[json.loads(l) for l in (ROOT/c['dataset_dir']/'train.metadata.jsonl').read_text(encoding='utf-8').splitlines()]
    lengths=[]
    for row in rows:
        ids,labels,prefix=encode_assistant_sample(tokenizer,row['messages'])
        weights=assistant_weights(tokenizer,row['messages'],ids,prefix,c['token_weighting'])
        assert labels[:prefix]==[-100]*prefix and len(weights)==len(ids)-prefix
        assert max(weights)==12 and ids[-1]==tokenizer.eos_token_id
        lengths.append(len(ids))
    assert len(lengths)==802 and max(lengths)<=c['max_length']


def test_real_tiny_phi3_lora_sparse_loss_and_gradients_match_full_masked_loss():
    import torch
    from transformers import Phi3Config,Phi3ForCausalLM
    from peft import LoraConfig,get_peft_model
    torch.manual_seed(7)
    base=Phi3ForCausalLM(Phi3Config(vocab_size=64,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
        max_position_embeddings=64,original_max_position_embeddings=64,
        pad_token_id=0,bos_token_id=1,eos_token_id=2,attention_dropout=0.,resid_pdrop=0.,embd_pdrop=0.))
    c=json.loads(CONFIG.read_text(encoding='utf-8'))
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=c['target_modules'],task_type='CAUSAL_LM'))
    model.train();model.config.use_cache=False
    ids=torch.randint(0,64,(1,12));labels=ids.clone();labels[:,:7]=-100
    batch={'input_ids':ids,'labels':labels,'attention_mask':torch.ones_like(ids)}
    weights=torch.tensor([[12.,3.,1.,1.,1.]])
    full=model(input_ids=ids,attention_mask=batch['attention_mask']).logits
    losses=torch.nn.functional.cross_entropy(full[:,6:-1].reshape(-1,64).float(),ids[:,7:].reshape(-1),reduction='none')
    reference=(losses*weights.reshape(-1)).sum()/weights.sum()
    reference.backward()
    gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    actual=completion_only_loss(model,batch,7,weights)
    actual.backward()
    assert torch.allclose(actual,reference,atol=1e-6)
    assert any(torch.count_nonzero(g) for g in gradients.values())
    for name,p in model.named_parameters():
        if p.requires_grad:assert torch.allclose(p.grad,gradients[name],atol=1e-6)
