"""Offline inference, one GPU model family at a time. Never trains or falls back to base."""
import argparse
from contextlib import redirect_stdout
import gc
import json
from pathlib import Path
import sys
import time

from scripts import qwen_worker
from src.training.a1_runtime import ROOT, file_hash


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_external(spec):
    base, adapter = ROOT / spec['model_dir'], ROOT / spec['adapter_dir']
    manifest = read(base / 'download_manifest.json')
    if manifest['repo'] != spec['base_model'] or manifest['revision'] != spec['revision']:
        raise ValueError('Sai model nền: ' + spec['base_model'])
    for name, sha in manifest['sha256'].items():
        if file_hash(base / name) != sha:
            raise ValueError('Sai checksum model: ' + name)
    config = read(adapter / 'adapter_config.json')
    if config['base_model_name_or_path'] != spec['base_model']:
        raise ValueError('Adapter khác model nền')
    imported = read(ROOT / 'outputs/teammate_adapter_import_20261111/verification.json')
    expected = next(row for row in imported if Path(row['directory']).resolve() == adapter.resolve())
    for name, sha in expected['hashes'].items():
        if file_hash(adapter / name) != sha:
            raise ValueError('Sai checksum adapter: ' + name)
    return {'repo': spec['base_model'], 'revision': spec['revision'], 'variant': 'finetuned',
            'adapter': str(adapter), 'sha256': file_hash(adapter / 'adapter_model.safetensors')}


class Worker:
    def __init__(self, config):
        self.config = config
        self.model = self.tokenizer = self.processor = self.dtype = None
        self.family = None
        self.identities = {}
        for key, role in [('synthesis', 'A4'), ('vision', 'A5')]:
            if config.get(key, {}).get('enabled', True) and key in config:
                self.identities[role] = verify_external(config[key])
        self.activate('qwen3')
        self.identity['agents'].update(self.identities)
        self.identity['image_agent'] = 'A5' in self.identities
        self.identity['execution'] = 'one_model_family_at_a_time'

    def release(self):
        import torch
        self.model = self.tokenizer = self.processor = None
        self.family = None
        gc.collect()
        torch.cuda.empty_cache()

    def activate(self, family):
        if self.family == family:
            return
        import torch
        self.release()
        if family == 'qwen3':
            self.model, self.tokenizer, self.dtype, identity = qwen_worker.load(self.config['text_models'])
            if not hasattr(self, 'identity'):
                self.identity = identity
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel, get_peft_model_state_dict
            spec = self.config[family]
            base, adapter = ROOT / spec['model_dir'], ROOT / spec['adapter_dir']
            self.dtype = torch.bfloat16
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=self.dtype)
            # New tokenizer exports store extra_special_tokens as a list; 4.57 expects a mapping.
            # All token IDs remain in the unmodified tokenizer.json.
            self.tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True, extra_special_tokens={})
            cls = AutoModelForCausalLM
            if family == 'vision':
                from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Qwen2VLImageProcessor, Qwen2VLVideoProcessor
                cls = Qwen2VLForConditionalGeneration
                # The image settings were saved in the newer combined processor format.
                processor_config = read(adapter / 'processor_config.json')
                image_config = processor_config['image_processor']
                image_processor = Qwen2VLImageProcessor.from_dict(image_config)
                video_processor = Qwen2VLVideoProcessor.from_dict(processor_config['video_processor'])
                self.processor = Qwen2VLProcessor(image_processor=image_processor, tokenizer=self.tokenizer,
                    video_processor=video_processor, chat_template=(adapter / 'chat_template.jinja').read_text(encoding='utf-8'))
            model = cls.from_pretrained(base, local_files_only=True, trust_remote_code=False,
                quantization_config=quant, device_map={'': 0}, dtype=self.dtype, attn_implementation='sdpa')
            # Transformers 4.57 uses model.language_model, matching the supplied vision adapter.
            self.model = PeftModel.from_pretrained(model, adapter, local_files_only=True, is_trainable=False)
            del model
            from safetensors.torch import load_file
            saved = load_file(str(adapter / 'adapter_model.safetensors'))
            loaded = get_peft_model_state_dict(self.model)
            if set(saved) != set(loaded) or any(not torch.equal(loaded[k].detach().cpu(),
                    saved[k].to(dtype=loaded[k].dtype)) for k in saved):
                raise ValueError('Adapter nạp thiếu hoặc sai trọng số: ' + str(adapter))
            del saved, loaded
            self.model.eval()
        self.family = family

    def generate(self, request):
        role = request['role']
        if role in ('A1', 'A2', 'A3'):
            self.activate('qwen3')
            return qwen_worker.generate(self.model, self.tokenizer, self.dtype,
                self.config['text_models'], request, self.identity)
        if role not in self.identities:
            raise ValueError('Agent chưa được cấu hình')
        import torch
        family = 'synthesis' if role == 'A4' else 'vision'
        self.activate(family)
        spec = self.config[family]
        if role == 'A5':
            from PIL import Image, ImageOps
            image_path = Path(request['image_path']).resolve()
            if not image_path.is_relative_to((ROOT / 'outputs/web_uploads').resolve()):
                raise ValueError('Ảnh phải được tải lên web trước')
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((512, 512))
            chat = [{'role': 'user', 'content': [{'type': 'image'},
                {'type': 'text', 'text': request['prompt']}]}]
            text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], return_tensors='pt').to('cuda')
        else:
            text = self.tokenizer.apply_chat_template(request['messages'], tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
        length = inputs['input_ids'].shape[1]
        if length > spec.get('max_input_tokens', 4096):
            raise ValueError(role + '_INPUT_TOO_LONG: cần rút gọn ngữ cảnh')
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=spec.get('max_new_tokens', 768),
                do_sample=False, temperature=None, top_p=None, top_k=None,
                eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
        raw = self.tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()
        if not raw:
            raise RuntimeError('Model trả nội dung rỗng')
        return {'id': request['id'], 'role': role, 'status': 'ok', 'raw_output': raw,
                'identity': self.identities[role], 'input_tokens': length,
                'output_tokens': output.shape[1] - length, 'generation_seconds': time.perf_counter() - start,
                'allocated_mib': round(torch.cuda.memory_allocated() / 1024**2, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            worker = Worker(read(args.config))
        print(json.dumps({'event': 'ready', 'status': 'ok', 'identity': worker.identity}), flush=True)
    except Exception as exc:
        print(json.dumps({'status': 'error', 'error': type(exc).__name__ + ': ' + str(exc)}), flush=True)
        return 1
    for line in sys.stdin:
        request = {}
        try:
            if len(line) > 100000:
                raise ValueError('Request vượt giới hạn')
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                result = worker.generate(request)
        except Exception as exc:
            result = {'id': request.get('id'), 'role': request.get('role'), 'status': 'error',
                      'error': type(exc).__name__ + ': ' + str(exc)}
            if 'worker' in locals():
                worker.release()
        print(json.dumps(result, ensure_ascii=False), flush=True)
    worker.release()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
