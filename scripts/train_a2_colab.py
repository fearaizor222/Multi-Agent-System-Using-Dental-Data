"""Portable Colab runner: continue A2 v3, fixed FP16, Drive checkpoints, no Modal."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time
import zipfile

from scripts.train_a2 import configure_tokenizer,encode_sample,parse_sample,score_answer,summarize_results,plot_results,inference_dtype_signature

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT_REQUIRED={'adapter_config.json','adapter_model.safetensors','trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def verify_bundle(root=ROOT):
    manifest=read(root/'bundle_manifest.json')
    for name,digest in manifest['files_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Bundle checksum mismatch: '+name)
    plan=read(root/'plan.json')
    raft=rows(root/'data/raft_train.jsonl');replay=rows(root/'data/replay.jsonl');validation=rows(root/'data/validation.jsonl')
    if (len(raft),len(replay),len(validation))!=(880,31,72):raise ValueError('Unexpected dataset counts')
    for record in raft+replay+validation:parse_sample(record)
    return plan,raft,replay,validation


def mixed_training_records(raft,replay,draws,seed):
    if not replay or draws<0:raise ValueError('Invalid replay configuration')
    order=list(range(len(replay)));rng=random.Random(seed);selected=[]
    while len(selected)<draws:
        rng.shuffle(order)
        selected.extend(replay[i] for i in order[:min(len(order),draws-len(selected))])
    result=raft+selected;rng.shuffle(result)
    return result


def tokenizer_for(plan):
    from transformers import AutoTokenizer
    return configure_tokenizer(AutoTokenizer.from_pretrained(plan['model'],revision=plan['revision'],trust_remote_code=False),'qwen3_17b')


def preflight(root=ROOT,tokenizer=None):
    plan,raft,replay,validation=verify_bundle(root)
    training=mixed_training_records(raft,replay,plan['replay_draws'],plan['seed'])
    tokenizer=tokenizer or tokenizer_for(plan)
    lengths={};encoded={}
    for name,records in [('train',training),('validation',validation)]:
        encoded[name]=[encode_sample(tokenizer,r,plan['max_length']) for r in records]
        lengths[name]={'count':len(records),'max_tokens':max(len(r['input_ids']) for r in encoded[name]),
            'max_assistant_tokens':max(sum(x!=-100 for x in r['labels']) for r in encoded[name])}
    if lengths['validation']['max_assistant_tokens']>plan['max_new_tokens']:raise ValueError('Generation cap too small')
    report={'ready':True,'raft_unique':len(raft),'replay_unique':len(replay),'replay_draws':plan['replay_draws'],
        'lengths':lengths,'precision':'fp16','initial_adapter_sha256':sha(root/'initial_adapter/adapter_model.safetensors'),
        'weights_loaded':False,'training_executed':False,'test_used':False}
    return report,plan,encoded,validation,tokenizer


def gpu_check():
    import torch
    from packaging.version import Version
    if not torch.cuda.is_available():raise RuntimeError('Chọn Runtime > Change runtime type > GPU trong Colab.')
    if torch.cuda.get_device_capability(0)[0]<6:raise RuntimeError('GPU này không hỗ trợ cấu hình NF4; cần GPU khác.')
    if Version(torch.__version__.split('+')[0])<Version('2.6'):
        raise RuntimeError('Cần PyTorch >=2.6 để resume checkpoint với Transformers này. Hãy dùng runtime Colab mới.')


def load_model(plan,adapter,*,trainable=False):
    import torch
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig
    from peft import PeftModel,prepare_model_for_kbit_training
    model=AutoModelForCausalLM.from_pretrained(plan['model'],revision=plan['revision'],trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16),
        device_map={'':0},dtype=torch.float16,attn_implementation='sdpa')
    if trainable:
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False})
    model=PeftModel.from_pretrained(model,str(adapter),is_trainable=trainable)
    if trainable:
        model.config.use_cache=False
        if not any(p.requires_grad for p in model.parameters()):raise RuntimeError('No trainable LoRA weights')
        if any(p.requires_grad and 'lora_' not in n for n,p in model.named_parameters()):raise RuntimeError('Base weights unexpectedly trainable')
    else:
        model.requires_grad_(False);model.eval();model.config.use_cache=True
    return model


def checkpoint_manifest(directory,step):
    if any(not (directory/n).is_file() for n in CHECKPOINT_REQUIRED):raise ValueError('Checkpoint is missing resume state')
    files={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name not in ('_complete.json','_complete.json.tmp')}
    save(directory/'_complete.json',{'step':step,'files_sha256':files})


def latest_complete_checkpoint(directory):
    candidates=sorted(directory.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]),reverse=True)
    for candidate in candidates:
        if not (candidate/'_complete.json').exists():continue
        manifest=read(candidate/'_complete.json')
        if CHECKPOINT_REQUIRED<=set(manifest['files_sha256']) and all((candidate/n).is_file() and sha(candidate/n)==digest for n,digest in manifest['files_sha256'].items()):
            return candidate
        print('Bỏ checkpoint chưa toàn vẹn:',candidate.name,flush=True)
    return None


def ensure_identity(run,plan):
    identity={'bundle_manifest_sha256':sha(ROOT/'bundle_manifest.json'),'plan':plan}
    path=run/'identity.json'
    if path.exists() and read(path)!=identity:raise ValueError('RUN_NAME đã thuộc bộ dữ liệu/cấu hình khác; hãy dùng tên mới.')
    save(path,identity)


def train(run):
    import torch
    from transformers import Trainer,TrainerCallback,TrainingArguments,DataCollatorForSeq2Seq,set_seed
    report,plan,encoded,validation,tokenizer=preflight()
    ensure_identity(run,plan)
    if (run/'training_complete.json').exists():
        print('Training đã hoàn tất. Chạy cell đánh giá sau train.',flush=True);return
    gpu_check();set_seed(plan['seed']);save(run/'preflight.json',report)
    checkpoint=latest_complete_checkpoint(run/'checkpoints')
    if checkpoint is None and any((run/'checkpoints').glob('checkpoint-*/_complete.json')):
        raise ValueError('Checkpoint đã lưu bị thiếu/hỏng dữ liệu. Không tự bắt đầu lại từ v3; hãy kiểm tra thư mục Drive.')
    print('Khôi phục checkpoint:' if checkpoint else 'Tiếp tục từ adapter A2 v3:',checkpoint or ROOT/'initial_adapter',flush=True)
    model=load_model(plan,ROOT/'initial_adapter',trainable=True)
    model.print_trainable_parameters()
    class Persist(TrainerCallback):
        def on_save(self,args,state,control,**kwargs):
            checkpoint_manifest(Path(args.output_dir)/f'checkpoint-{state.global_step}',state.global_step)
            save(run/'training_log.json',state.log_history)
    args=TrainingArguments(output_dir=str(run/'checkpoints'),num_train_epochs=plan['epochs'],
        per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,
        learning_rate=plan['learning_rate'],warmup_ratio=.05,fp16=True,bf16=False,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        eval_strategy='steps',save_strategy='steps',eval_steps=25,save_steps=25,logging_steps=5,
        save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,
        optim='adamw_torch',report_to='none',seed=plan['seed'],data_seed=plan['seed'],label_names=['labels'])
    trainer=Trainer(model=model,args=args,train_dataset=encoded['train'],eval_dataset=encoded['validation'],
        data_collator=DataCollatorForSeq2Seq(tokenizer,label_pad_token_id=-100,pad_to_multiple_of=8),
        processing_class=tokenizer,callbacks=[Persist()])
    save(run/'run_status.json',{'status':'training','resume_checkpoint':str(checkpoint) if checkpoint else None,
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'precision':'fp16'})
    try:
        result=trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        trainer.save_model(str(run/'adapter'));tokenizer.save_pretrained(run/'adapter')
        save(run/'training_log.json',trainer.state.log_history)
        save(run/'train_metrics.json',result.metrics)
        save(run/'training_complete.json',{'best_checkpoint':trainer.state.best_model_checkpoint,
            'best_validation_loss':trainer.state.best_metric,'global_step':trainer.state.global_step,
            'adapter_sha256':sha(run/'adapter/adapter_model.safetensors'),'optimizer_state_resumed':checkpoint is not None})
        save(run/'run_status.json',{'status':'trained_pending_evaluation'})
    except BaseException as exc:
        save(run/'run_status.json',{'status':'interrupted_or_failed','error':str(exc)[:1500]});raise
    finally:
        del trainer,model;gc.collect();torch.cuda.empty_cache()


def evaluate_stage(run,stage):
    import torch
    from transformers import set_seed
    plan,_,_,records=verify_bundle();ensure_identity(run,plan)
    adapter=ROOT/'initial_adapter' if stage=='before' else run/'adapter'
    if stage=='after' and not (run/'training_complete.json').exists():raise ValueError('Training chưa hoàn tất.')
    directory=run/stage;directory.mkdir(parents=True,exist_ok=True)
    identity={'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'validation_sha256':sha(ROOT/'data/validation.jsonl'),
        'precision':'fp16','thinking':False,'do_sample':False,'max_new_tokens':plan['max_new_tokens']}
    if (directory/'identity.json').exists() and read(directory/'identity.json')!=identity:raise ValueError('Evaluation output belongs to different weights/settings')
    save(directory/'identity.json',identity)
    path=directory/'predictions.jsonl';predictions=rows(path) if path.exists() else []
    if len(predictions)>len(records):raise ValueError('Too many saved predictions')
    for record,pred in zip(records,predictions):
        if record['metadata']['id']!=pred['sample_id']:raise ValueError('Saved prediction order mismatch')
        score=score_answer(record,pred['raw_output'])
        if any(pred[k]!=v for k,v in score.items()):raise ValueError('Saved prediction score mismatch')
    if len(predictions)<len(records):
        gpu_check();set_seed(plan['seed']);tokenizer=tokenizer_for(plan)
        model=load_model(plan,adapter)
        save(directory/'dtype_signature.json',inference_dtype_signature(model))
        try:
            for record in records[len(predictions):]:
                ids=tokenizer.apply_chat_template(record['messages'][:2],tokenize=True,add_generation_prompt=True,
                    enable_thinking=False,return_tensors='pt').to('cuda')
                started=time.monotonic()
                with torch.inference_mode():
                    result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                        temperature=None,top_p=None,top_k=None,max_new_tokens=plan['max_new_tokens'],
                        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
                raw=tokenizer.decode(result[0,ids.shape[1]:],skip_special_tokens=True).strip()
                predictions.append({'sample_id':record['metadata']['id'],'raw_output':raw,
                    'prompt_ids_sha256':hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                    'seconds':time.monotonic()-started,'generation_tokens':int(result.shape[1]-ids.shape[1]),
                    **score_answer(record,raw)})
                temporary=path.with_suffix('.tmp');temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions),encoding='utf-8');os.replace(temporary,path)
                print(f'{stage}: {len(predictions)}/{len(records)}',flush=True)
        finally:
            del model;gc.collect();torch.cuda.empty_cache()
    metrics=summarize_results(predictions);save(directory/'metrics.json',metrics)
    if stage=='after' and (run/'before/metrics.json').exists():
        plot_results(read(run/'training_log.json'),read(run/'before/metrics.json'),metrics,run/'figures',
            model_name='Qwen3-1.7B: A2 v3 -> RAFT v2')
        save(run/'run_status.json',{'status':'completed','semantic_review_still_required':True,'test_evaluated':False})
    print(json.dumps(metrics,indent=2),flush=True)


def export(run):
    if not all((run/name).exists() for name in ('training_complete.json','before/metrics.json','after/metrics.json')):
        raise ValueError('Cần hoàn tất train và hai lượt đánh giá trước khi xuất kết quả cuối.')
    destination=run.parent/(run.name+'_results.zip')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.rglob('*')):
            if path.is_file() and 'checkpoints' not in path.relative_to(run).parts and path.suffix!='.tmp':
                archive.write(path,str(path.relative_to(run)).replace('\\','/'))
    print('Results:',destination,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['preflight','before','train','after','export'],required=True)
    parser.add_argument('--output-dir',type=Path)
    options=parser.parse_args()
    if options.stage=='preflight':print(json.dumps(preflight()[0],ensure_ascii=False,indent=2))
    else:
        if options.output_dir is None:parser.error('--output-dir is required')
        options.output_dir.mkdir(parents=True,exist_ok=True)
        if options.stage=='train':train(options.output_dir)
        elif options.stage=='export':export(options.output_dir)
        else:evaluate_stage(options.output_dir,options.stage)
