"""Inference smoke checks for installed adapters; no training or benchmark selection."""
import json
from pathlib import Path
import subprocess
from src.core.shared_qwen import SharedQwenClient
from src.core.a1_process import ROOT


def main():
    client = SharedQwenClient(ROOT / 'config/assistant_local.json')
    folder = ROOT / 'outputs/integration_smoke_20261111'
    folder.mkdir(exist_ok=True)
    result = {}
    try:
        result['identity'] = client.ensure_started()
        print('Worker ready: A1/A2/A3 adapters, synthesis and vision registered', flush=True)
        result['A1'] = client.plan({'message': 'Xin chào', 'history': [], 'attachments': [],
            'capabilities': {'A2': True, 'A5': False, 'lookup_clinic_service': False}})
        print('A1 generated', flush=True)
        for role in ('A2', 'A3'):
            run = ROOT / client.config[role.lower() + '_run_dir']
            sample = json.loads((run / 'validation.jsonl').read_text(encoding='utf-8').splitlines()[0])
            result[role] = client.generate(role, sample['messages'][:2])
            print(role, 'generated', flush=True)
        from src.agents.teammate_synthesis import SYSTEM
        payload = {'question': 'Máng nhai là gì?', 'approved_claims': [
            {'claim_id': 'c1', 'text': 'Máng nhai là một khí cụ bằng nhựa.', 'citations': ['I1']}]}
        result['A4'] = client.generate('A4', [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}])
        print('Synthesis generated', flush=True)
        # Copy an existing repository image as a test fixture, without modifying its pixels.
        blob = subprocess.run(['git', 'show', 'origin/main:data/images/europe_pmc/PMC3425398_10.1177_1758736012456543-fig1.jpg'],
            cwd=ROOT, check=True, stdout=subprocess.PIPE).stdout
        image = ROOT / 'outputs/web_uploads/smoke_repo_image.jpg'
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(blob)
        from src.agents.local_vision import analyze_image
        result['A5'] = analyze_image(client, image, 'Mô tả những gì có thể quan sát được trong ảnh.')
        print('Vision generated', flush=True)
        # Ensure the shared Qwen adapters still work after unloading two other model families.
        result['A1_after_switch'] = client.plan({'message': 'Cảm ơn', 'history': [], 'attachments': [],
            'capabilities': {'A2': True, 'A5': False, 'lookup_clinic_service': False}})
        print('Qwen3 reloaded after synthesis and vision', flush=True)
    finally:
        (folder / 'model_calls.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
