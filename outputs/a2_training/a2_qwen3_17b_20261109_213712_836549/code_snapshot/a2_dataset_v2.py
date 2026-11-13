"""Validate immutable A2 v2 splits and proof provenance; exclude test from training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def load_v2(directory, parse_sample):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest['version'].startswith('a2-v2-grounded-'):
        raise ValueError('Unsupported A2 v2 manifest')
    packet = {}; hashes = {}; ids = set(); inputs = set(); groups = {}; chunks = {}
    for name in ('train', 'val', 'test'):
        files = {}
        for suffix in ('.jsonl', '.metadata.jsonl'):
            filename = name + suffix
            raw = (directory / filename).read_bytes()
            hashes[filename] = hashlib.sha256(raw).hexdigest()
            expected_sha = manifest.get('files_sha256', {}).get(filename)
            if expected_sha and expected_sha != hashes[filename]:
                raise ValueError('Dataset checksum mismatch: ' + filename)
            files[suffix] = [json.loads(l) for l in raw.decode('utf-8-sig').splitlines() if l.strip()]
        records, metas = files['.jsonl'], files['.metadata.jsonl']
        if not records or len(records) != len(metas):
            raise ValueError('Empty or mismatched split ' + name)
        groups[name] = set(); chunks[name] = set(); merged = []
        for row, meta in zip(records, metas):
            payload, gold = parse_sample(row)
            if (meta['split'] != name or meta['id'] in ids or meta['question'] != payload['question']
                    or meta['status'] != gold['status'] or len(meta['chunk_ids']) != len(payload['evidence'])):
                raise ValueError('Invalid A2 v2 metadata')
            ids.add(meta['id'])
            signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if signature in inputs:
                raise ValueError('Duplicate input')
            inputs.add(signature)
            groups[name].add(meta['document_id']); chunks[name].update(meta['chunk_ids'])
            review = meta['review']
            approved = (review['clear'] and review.get('actual_case')=='missing_evidence' and not gold['claims']) if gold['status']=='insufficient_evidence' else all(review[k] for k in
                    ('clear','all_claims_supported','answers_current_question','supported_part_covered','citations_correct'))
            if review['status'] != gold['status'] or not approved:
                raise ValueError('Unapproved reference answer')
            if gold['status']=='insufficient_evidence':
                blind = meta.get('blind_answerability', {})
                if not blind.get('decision_unambiguous') or blind.get('status')!='insufficient_evidence' or blind.get('answerable_quotes'):
                    raise ValueError('Missing blind answerability approval')
            if len(meta['proofs']) != len(gold['claims']):
                raise ValueError('Missing claim proofs')
            sources = {e['evidence_id']: ' '.join(e['content'].casefold().split()) for e in payload['evidence']}
            for claim, proofclaim in zip(gold['claims'], meta['proofs']):
                if claim != {k:proofclaim[k] for k in ('text','evidence_ids')}:
                    raise ValueError('Proof claim mismatch')
                for proof in proofclaim['proofs']:
                    if ' '.join(proof['quote'].casefold().split()) not in sources[proof['evidence_id']]:
                        raise ValueError('Non-source quote')
            merged.append({**row, 'metadata': meta})
        packet['validation' if name == 'val' else name] = merged
    for a, b in (('train','val'),('train','test'),('val','test')):
        if groups[a] & groups[b] or chunks[a] & chunks[b]:
            raise ValueError('Cross-split source overlap')
    for filename, sha in hashes.items():
        if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != sha:
            raise ValueError('Dataset changed during validation')
    packet['info'] = {'dataset_version': manifest['version'], 'source_sha256': hashes,
        'split_method': manifest['split_method'], 'counts': {s:len(rows) for s,rows in packet.items()},
        'labels': {s:dict(Counter(r['metadata']['status'] for r in rows)) for s,rows in packet.items()},
        'test_reserved': True, 'cross_split_document_and_chunk_overlap': 0,
        'scope': manifest['scope']}
    # Metadata records test size/hash only: no test prompts or labels enter the trainer.
    packet.pop('test')
    return packet
