"""Local tokenizer and tiny CPU architecture tests; no downloaded weights/training run."""
import json

import pytest

from scripts.check_a1_environment import completion_only_loss, encode_assistant_sample
from src.training.a1_model_support import configure_tokenizer, load_tokenizer, tokenizer_identity
from src.training.a1_runtime import ROOT
from src.training.a1_strategy import assistant_weights


def config():
    return json.loads((ROOT / "config/a1_vinallama_27b_v4.json").read_text(encoding="utf-8"))


def test_vinallama_preserves_experiment_settings_and_dataset():
    old = json.loads((ROOT / "config/a1_qwen3_17b_v4.json").read_text(encoding="utf-8"))
    new = config()
    allowed = {"purpose", "base_model", "revision", "model_dir", "chat_format"}
    assert {k:v for k,v in old.items() if k not in allowed} == {k:v for k,v in new.items() if k not in allowed}


def test_chatml_has_exact_assistant_prefix_and_eos_without_qwen_thinking():
    c = config()
    tokenizer = load_tokenizer(c, ROOT)
    messages = [{"role":"system","content":"policy"}, {"role":"user","content":"question"},
                {"role":"assistant","content":'{"decision":"respond"}'}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert text == '<|im_start|>system\npolicy<|im_end|>\n<|im_start|>user\nquestion<|im_end|>\n<|im_start|>assistant\n{"decision":"respond"}<|im_end|>'
    ids, labels, prefix = encode_assistant_sample(tokenizer, messages)
    assert ids[-1] == tokenizer.eos_token_id == 46304
    assert all(value == -100 for value in labels[:prefix])
    assert labels[prefix:] == ids[prefix:]
    assert "<think>" not in text
    assert tokenizer_identity(tokenizer, c)["chat_format"] == "vinallama_chatml_v1"


def test_sentencepiece_weights_use_exact_context_offsets_and_leave_eos_at_one():
    c = config()
    tokenizer = load_tokenizer(c, ROOT)
    row = json.loads((ROOT/c["dataset_dir"]/"train.metadata.jsonl").read_text(encoding="utf-8").splitlines()[0])
    messages = row["messages"]
    ids, labels, prefix = encode_assistant_sample(tokenizer, messages)
    raw = messages[-1]["content"]
    assert ids[prefix:prefix+len(tokenizer(raw,add_special_tokens=False)["input_ids"])] != tokenizer(raw,add_special_tokens=False)["input_ids"]
    weights = assistant_weights(tokenizer, messages, ids, prefix, c["token_weighting"])
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
    encoding = tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    assert encoding["input_ids"] == ids
    begin=text.rfind(raw)
    decision_start=begin+len('{"decision":"')
    decision_end=begin+raw.index('"',len('{"decision":"'))
    control_end=begin+raw.index(',"resolved_request":')
    expected=[]
    for left,right in encoding["offset_mapping"][prefix:]:
        expected.append(12. if left<decision_end and right>decision_start else
                        3. if right>begin and left<control_end else 1.)
    assert weights == expected and max(weights)==12. and weights[-1]==1.
    bad=list(ids); bad[prefix]=tokenizer.unk_token_id
    with pytest.raises(ValueError,match="không khớp"):
        assistant_weights(tokenizer,messages,bad,prefix,c["token_weighting"])


def test_vinallama_requires_explicit_format():
    c=config()
    c.pop("chat_format")
    with pytest.raises(ValueError,match="chat_format"):
        configure_tokenizer(None,c)


def test_qwen_native_template_and_generation_tokens_unchanged():
    from transformers import AutoTokenizer
    c=json.loads((ROOT/'config/a1_qwen3_17b_v4.json').read_text(encoding='utf-8'))
    before=AutoTokenizer.from_pretrained(ROOT/c['model_dir'],local_files_only=True)
    after=load_tokenizer(c,ROOT)
    assert before.chat_template==after.chat_template
    assert before.eos_token_id==after.eos_token_id
    assert before.pad_token_id==after.pad_token_id


def test_real_tiny_llama_lora_selective_logits_loss_and_gradients_match_full_loss():
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import LlamaConfig,LlamaForCausalLM
    torch.manual_seed(12)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.))
    model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,target_modules=config()['target_modules'],task_type='CAUSAL_LM',lora_dropout=0.))
    model.train(); model.config.use_cache=False
    ids=torch.randint(0,64,(1,12)); labels=ids.clone(); labels[:,:7]=-100
    batch={'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels}
    weights=torch.tensor([[12.,3.,1.,1.,1.]])
    logits=model(input_ids=ids,attention_mask=batch['attention_mask']).logits
    reference_losses=torch.nn.functional.cross_entropy(logits[:,6:-1].reshape(-1,64).float(),ids[:,7:].reshape(-1),reduction='none')
    reference=(reference_losses*weights.reshape(-1)).sum()/weights.sum()
    reference.backward()
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    actual=completion_only_loss(model,batch,7,weights)
    actual.backward()
    assert torch.allclose(actual,reference,atol=1e-6)
    assert any(torch.count_nonzero(value) for value in grads.values())
    for name,p in model.named_parameters():
        if p.requires_grad:
            assert torch.isfinite(p.grad).all()
            assert torch.allclose(p.grad,grads[name],atol=1e-6)
