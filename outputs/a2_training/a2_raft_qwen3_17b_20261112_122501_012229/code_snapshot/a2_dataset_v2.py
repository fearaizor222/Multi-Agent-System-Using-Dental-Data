"""Validate immutable A2 v2 splits and proof provenance; exclude test from training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def validate_v3_provenance(row, meta, payload, review_log, blind_log, natural_log=None):
    actual=canonical_sha(row['messages'])
    review=meta.get('review_record',{})
    if meta.get('binding_sha256')!=actual or review.get('binding_sha256')!=actual:
        raise ValueError('Stale review binding: messages changed')
    if canonical_sha(review) not in review_log or review.get('model')!='gemini-3.1-flash-lite':
        raise ValueError('Missing independent review log entry')
    packet={'id':meta['id'],'question':payload['question'],
            'history':payload.get('conversation',{}).get('history',[]),
            'evidence':payload['evidence'],'proposed_claims':meta['proofs']}
    negative=meta['status']=='insufficient_evidence'
    if negative:
        blind=meta.get('blind_record',{})
        if (canonical_sha(blind) not in blind_log or blind.get('model')!='gemini-3.5-flash-lite'
                or blind.get('binding_sha256')!=actual):
            raise ValueError('Missing real blind review log entry')
        packet.pop('proposed_claims')
    if review.get('request_sha256')!=canonical_sha(packet):
        raise ValueError('Review request does not match final question/answer')
    raw=review.get('result',{})
    if (raw.get('id')!=meta['id'] or not raw.get('decision_unambiguous')
            or raw.get('status')!=meta['status']):
        raise ValueError('Unapproved fresh review')
    if not negative and not raw.get('question_natural'):
        raise ValueError('Unnatural question in fresh review')
    if meta['status']=='draft' and not all(raw.get(k) for k in
           ('all_claims_supported','answers_current_question','supported_part_covered','citations_correct','no_source_conflict')):
        raise ValueError('Unsupported answer in fresh review')
    if negative:
        if blind.get('request_sha256')!=canonical_sha(packet):
            raise ValueError('Blind request contains stale or non-blind input')
        result=blind.get('result',{})
        if result!=meta.get('blind_answerability') or result.get('status')!='insufficient_evidence' or not result.get('decision_unambiguous') or result.get('answerable_quotes'):
            raise ValueError('Blind answerability disagreement')
        if raw.get('answerable_quotes'):
            raise ValueError('Independent negative reviewer found answerable evidence')
        natural=meta.get('natural_question_review',{})
        if canonical_sha(natural) not in (natural_log or set()) or natural.get('binding_sha256')!=actual:
            raise ValueError('Missing exact-input question-coherence review')
        expected={'id':meta['id'],'question':payload['question'],'history':payload.get('conversation',{}).get('history',[])}
        if natural.get('request_sha256')!=canonical_sha(expected) or not natural.get('result',{}).get('question_natural'):
            raise ValueError('Unnatural or stale question-coherence review')


def load_v2(directory, parse_sample):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['version'].startswith('a2-v4-audited-'):
        return load_audited_v4(directory, manifest, parse_sample)
    if not manifest['version'].startswith(('a2-v2-grounded-','a2-v3-repaired-')):
        raise ValueError('Unsupported A2 v2 manifest')
    is_v3=manifest['version'].startswith('a2-v3-')
    review_log=set();blind_log=set();natural_log=set()
    if is_v3:
        for name in ('review','blind','negative_review','natural_negatives'):
            path=directory/'audit'/(name+'.jsonl')
            expected=manifest['audit_sha256'][name+'.jsonl']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Audit log checksum mismatch')
            entries={canonical_sha(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}
            if name in ('review','negative_review'):review_log.update(entries)
            elif name=='blind':blind_log=entries
            else:natural_log=entries
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
            if is_v3:validate_v3_provenance(row,meta,payload,review_log,blind_log,natural_log)
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
        'audit_sha256': manifest.get('audit_sha256',{}),
        'split_method': manifest['split_method'], 'counts': {s:len(rows) for s,rows in packet.items()},
        'labels': {s:dict(Counter(r['metadata']['status'] for r in rows)) for s,rows in packet.items()},
        'test_reserved': True, 'cross_split_document_and_chunk_overlap': 0,
        'scope': manifest['scope']}
    # Metadata records test size/hash only: no test prompts or labels enter the trainer.
    packet.pop('test')
    return packet


def load_audited_v4(directory, manifest, parse_sample):
    """Verify manual patches against the fully validated v3 parent and exact change log."""
    parent = (directory / manifest['parent_directory']).resolve()
    if parent == directory.resolve():
        raise ValueError('Dataset cannot be its own parent')
    parent_manifest = json.loads((parent/'manifest.json').read_text(encoding='utf-8'))
    if not parent_manifest['version'].startswith('a2-v3-repaired-'):
        raise ValueError('Audited v4 requires a v3 parent')
    load_v2(parent, parse_sample)
    for name, checksum in manifest['parent_files_sha256'].items():
        if hashlib.sha256((parent/name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Audited dataset parent changed: '+name)
    for name, checksum in manifest['files_sha256'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != checksum:
            raise ValueError('Audited dataset checksum mismatch: '+name)
    read = lambda path: [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    changes = read(directory/'changes.jsonl')
    changes_by_id = {item['id']:item for item in changes}
    if len(changes_by_id) != len(changes):
        raise ValueError('Duplicate manual change')
    if dict(Counter(c['action'] for c in changes)) != manifest['actions']:
        raise ValueError('Manual action counts mismatch')
    expected, expected_meta, used = [], [], set()
    for row, meta in zip(read(parent/'train.jsonl'), read(parent/'train.metadata.jsonl')):
        change = changes_by_id.get(meta['id'])
        if change is None:
            expected.append(row); expected_meta.append(meta)
            continue
        used.add(meta['id'])
        if (change['messages_sha256'] != canonical_sha(row['messages'])
                or change['metadata_sha256'] != canonical_sha(meta)):
            raise ValueError('Patch bound to a different parent sample')
        if change['action'] == 'quarantine':
            continue
        if change['action'] != 'repair' or not change.get('reviewer','').startswith('assistant source-reading'):
            raise ValueError('Missing manual source-reading provenance')
        after, after_meta = change['after_record'], change['after_metadata']
        if (canonical_sha(after['messages']) != change['after_messages_sha256']
                or canonical_sha(after_meta) != change['after_metadata_sha256']
                or after_meta['id'] != meta['id'] or after_meta['family_id'] != meta['family_id']
                or after_meta['document_id'] != meta['document_id']):
            raise ValueError('Manual patch fingerprint or family mismatch')
        expected.append(after); expected_meta.append(after_meta)
    if used != set(changes_by_id):
        raise ValueError('Manual patch references unknown sample')
    if expected != read(directory/'train.jsonl') or expected_meta != read(directory/'train.metadata.jsonl'):
        raise ValueError('Train does not match reviewed patches')
    packet, documents, chunks, families, questions = {}, {}, {}, {}, {}
    seen_ids = set()
    for split in ('train','val','test'):
        if split != 'train':
            for suffix in ('.jsonl','.metadata.jsonl'):
                if (directory/(split+suffix)).read_bytes() != (parent/(split+suffix)).read_bytes():
                    raise ValueError('Reserved split changed: '+split)
        records, metas = read(directory/(split+'.jsonl')), read(directory/(split+'.metadata.jsonl'))
        if len(records) != len(metas) or len(records) != manifest['counts'][split]:
            raise ValueError('Split counts mismatch')
        documents[split], chunks[split], families[split], questions[split] = set(), set(), set(), set()
        merged = []
        for row, meta in zip(records, metas):
            payload, gold = parse_sample(row)
            if (meta['id'] in seen_ids or meta['split'] != split or meta['question'] != payload['question']
                    or meta['status'] != gold['status'] or len(meta['chunk_ids']) != len(payload['evidence'])
                    or meta['claims_count'] != len(gold['claims'])
                    or canonical_sha(row['messages']) != meta['binding_sha256']):
                raise ValueError('Invalid patched dataset metadata')
            seen_ids.add(meta['id'])
            if len(meta['proofs']) != len(gold['claims']):
                raise ValueError('Missing proof')
            sources = {e['evidence_id']:' '.join(e['content'].split()) for e in payload['evidence']}
            for claim, proof_claim in zip(gold['claims'],meta['proofs']):
                if claim != {k:proof_claim[k] for k in ('text','evidence_ids')}:
                    raise ValueError('Claim/proof mismatch')
                if set(claim['evidence_ids']) != {q['evidence_id'] for q in proof_claim['proofs']}:
                    raise ValueError('Claim citations lack proof coverage')
                for q in proof_claim['proofs']:
                    if not q['quote'].strip() or ' '.join(q['quote'].casefold().split()) not in sources[q['evidence_id']].casefold():
                        raise ValueError('Quote not present in source: '+meta['id'])
            documents[split].add(meta['document_id']); chunks[split].update(meta['chunk_ids'])
            families[split].add(meta['family_id']); questions[split].add(' '.join(payload['question'].casefold().split()))
            merged.append({**row,'metadata':meta})
        packet['validation' if split=='val' else split] = merged
    for a,b in (('train','val'),('train','test'),('val','test')):
        if any(groups[a] & groups[b] for groups in (documents,chunks,families,questions)):
            raise ValueError('Cross-split overlap')
    packet.pop('test')
    packet['info'] = {'dataset_version':manifest['version'],'source_sha256':manifest['files_sha256'],
        'counts':manifest['counts'],'test_reserved':True,'cross_split_document_and_chunk_overlap':0,
        'scope':manifest['scope'],'manual_actions':manifest['actions']}
    return packet
