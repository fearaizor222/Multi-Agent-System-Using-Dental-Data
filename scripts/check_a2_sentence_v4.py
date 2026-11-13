"""Verification checks for Stage 2 (v4) `data/processed/a2_sentence_selection_v4`."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

from src.training.a2_sentence_protocol import SentenceDraft, resolve_selection

ROOT = Path(__file__).resolve().parents[1]
V4_DIR = ROOT / 'data/processed/a2_sentence_selection_v4'

FORBIDDEN_ARTIFACTS = (
    '\xad', '\x07', '\ue023', 'ƣ', 'Ƣ', 'B=)', 'LMT?Đ', 'Snpee',
    'lỗi cầu', 'hàm đưới', 'bản lễ', 'bản lâ', 'lỗng múi', 'lỗổng múi',
    'cần trở', 'Cơ chân bướm rong', 'tối tứu', 'lối %u', 'mặt phẳng hgang',
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _parse_user_payload(rec: dict) -> dict:
    user_text = rec['messages'][1]['content']
    payload_str, schema_str = user_text.split('\nJSON schema:\n', 1)
    schema = json.loads(schema_str)
    claim_props = schema['$defs']['SelectedClaim']['properties']
    if list(claim_props.keys()) != ['evidence_ids', 'sentence_ids', 'text']:
        raise AssertionError(f'Unexpected schema claim property order: {list(claim_props.keys())}')
    if claim_props['evidence_ids']['items'].get('pattern') != r'^I\d+$':
        raise AssertionError('Missing evidence_ids regex pattern in user schema')
    if claim_props['sentence_ids']['items'].get('pattern') != r'^I\d+:S\d+$':
        raise AssertionError('Missing sentence_ids regex pattern in user schema')
    return json.loads(payload_str)


def _reconstruct_raw_payload(u_payload: dict) -> dict:
    """Verify segmented sentence offsets match raw units without touching hidden files."""
    return {
        'question': u_payload['question'],
        'evidence': [
            {
                'evidence_id': e['evidence_id'],
                'title': e['title'],
                'content': '\n'.join(s['text'] for s in e['sentences']),
            }
            for e in u_payload['evidence']
        ],
    }


def check(v4_dir: Path = V4_DIR, write_report: bool = True) -> dict:
    manifest = json.loads((v4_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('test_examples_used') is not False:
        raise AssertionError('manifest.test_examples_used must be False')

    seen_prompts: dict[str, str] = {}
    docs_by_split: dict[str, set[str]] = {}
    summary: dict[str, dict] = {}

    for split in ('train', 'val', 'challenge'):
        records = _read_jsonl(v4_dir / f'{split}.jsonl')
        metas = _read_jsonl(v4_dir / f'{split}.metadata.jsonl')
        if len(records) != len(metas):
            raise AssertionError(f'{split}: record/meta count mismatch ({len(records)} vs {len(metas)})')

        split_docs: set[str] = set()
        statuses = Counter()
        ev_counts = Counter()
        kinds = Counter()
        v4_intact_pos = Counter()
        partial_dirs = Counter()
        connectors = Counter()

        for rec, meta in zip(records, metas):
            u_payload = _parse_user_payload(rec)
            assistant = json.loads(rec['messages'][-1]['content'])
            draft = SentenceDraft.model_validate(assistant)

            for claim in assistant['claims']:
                if list(claim.keys()) != ['evidence_ids', 'sentence_ids', 'text']:
                    raise AssertionError(f'{split}/{meta["id"]}: claim key order {list(claim.keys())}')

            # Verify every segmented sentence has alphanumeric content and no forbidden OCR residue.
            for ev in u_payload['evidence']:
                for sent in ev['sentences']:
                    if not any(ch.isalnum() for ch in sent['text']):
                        raise AssertionError(f'{split}/{meta["id"]}: empty/punct-only unit {sent!r}')
                    for bad in FORBIDDEN_ARTIFACTS:
                        if bad in sent['text']:
                            raise AssertionError(f'{split}/{meta["id"]}: residual artifact {bad!r} in {sent["text"]!r}')

            for bad in FORBIDDEN_ARTIFACTS:
                if bad in u_payload['question']:
                    raise AssertionError(f'{split}/{meta["id"]}: residual artifact {bad!r} in question')

            # Verify unit IDs match exact sentence order in payload.
            valid_map = {
                s['sentence_id']: (ev['evidence_id'], s['text'])
                for ev in u_payload['evidence']
                for s in ev['sentences']
            }
            for claim in draft.claims:
                if any(sid not in valid_map for sid in claim.sentence_ids):
                    raise AssertionError(f'{split}/{meta["id"]}: unknown sentence_id {claim.sentence_ids}')
                if set(claim.evidence_ids) != {valid_map[sid][0] for sid in claim.sentence_ids}:
                    raise AssertionError(f'{split}/{meta["id"]}: evidence/sentence mismatch')

            # Prompt uniqueness check within & across splits.
            p_key = json.dumps(rec['messages'][:2], ensure_ascii=False, sort_keys=True)
            a_str = rec['messages'][-1]['content']
            if p_key in seen_prompts:
                raise AssertionError(f'{split}/{meta["id"]}: duplicate prompt detected')
            seen_prompts[p_key] = a_str

            # Collect docs (both primary and distractors).
            if 'document_ids' in meta:
                split_docs.update(meta['document_ids'])
            elif 'document_id' in meta:
                split_docs.add(meta['document_id'])
            for ddoc in meta.get('distractor_documents', []):
                split_docs.add(ddoc)

            statuses[draft.status] += 1
            ev_counts[len(u_payload['evidence'])] += 1
            kinds[meta['kind']] += 1
            if meta.get('variant', '').startswith('v4_intact'):
                v4_intact_pos[meta['owner_evidence_id']] += 1
                if meta['kind'] == 'intact_partial_support':
                    partial_dirs[meta['partial_direction']] += 1
                if meta.get('connector') and meta['connector'] not in ('none', 'single'):
                    connectors[meta['connector']] += 1

        docs_by_split[split] = split_docs
        if ev_counts[1] == 0 or ev_counts[2] == 0 or (split != 'val' and ev_counts[3] == 0):
            if split == 'val':
                # Val mirrors v2_reviewed 4 families (no artificial 3-ev alteration needed if v2 had 2/3)
                if not ev_counts:
                    raise AssertionError('val has empty ev_counts')
            else:
                raise AssertionError(f'{split}: missing 1/2/3 evidence mix: {dict(ev_counts)}')

        if split != 'val':
            if partial_dirs['pos_first'] != partial_dirs['neg_first'] or partial_dirs['pos_first'] == 0:
                raise AssertionError(f'{split}: partial_direction imbalance: {dict(partial_dirs)}')
            if len(connectors) < 4:
                raise AssertionError(f'{split}: insufficient connector diversity: {dict(connectors)}')
            max_conn = max(connectors.values())
            if max_conn / sum(connectors.values()) > 0.35:
                raise AssertionError(f'{split}: connector dominance: {dict(connectors)}')
            if not ({'I1', 'I2', 'I3'} <= set(v4_intact_pos.keys())):
                raise AssertionError(f'{split}: v4_intact gold missing I1/I2/I3 spread: {dict(v4_intact_pos)}')

        summary[split] = {
            'count': len(records),
            'statuses': dict(statuses),
            'evidence_counts': dict(sorted(ev_counts.items())),
            'kinds': dict(kinds),
            'v4_intact_gold_positions': dict(sorted(v4_intact_pos.items())),
            'partial_directions': dict(partial_dirs),
            'connectors': dict(connectors),
            'documents': sorted(split_docs),
        }

    # Document disjointness across train, val, challenge.
    if docs_by_split['train'] & docs_by_split['val']:
        raise AssertionError(f"train/val doc overlap: {docs_by_split['train'] & docs_by_split['val']}")
    if docs_by_split['train'] & docs_by_split['challenge']:
        raise AssertionError(f"train/challenge doc overlap: {docs_by_split['train'] & docs_by_split['challenge']}")
    if docs_by_split['val'] & docs_by_split['challenge']:
        raise AssertionError(f"val/challenge doc overlap: {docs_by_split['val'] & docs_by_split['challenge']}")

    # Verify bottle_label decimal integrity & OHA-CH01 bullet integrity in val/challenge.
    val_records = _read_jsonl(v4_dir / 'val.jsonl')
    val_text = '\n'.join(
        s['text'] for r in val_records for e in _parse_user_payload(r)['evidence'] for s in e['sentences']
    )
    if '>1.4 mg fluoride/L' not in val_text or '53.8–58.3' not in val_text:
        raise AssertionError('bottle_label decimal unit broken in v4 val')
    for r in val_records:
        for e in _parse_user_payload(r)['evidence']:
            for s in e['sentences']:
                if s['text'].endswith('>1.4') or s['text'].endswith('>0.8'):
                    raise AssertionError(f'Split decimal unit still exists in val: {s["text"]!r}')

    chal_records = _read_jsonl(v4_dir / 'challenge.jsonl')
    for r in chal_records:
        for e in _parse_user_payload(r)['evidence']:
            for s in e['sentences']:
                if '■' in s['text'][1:]:
                    raise AssertionError(f'Fused ■ bullet still exists in challenge: {s["text"]!r}')

    if write_report:
        (v4_dir / 'verification.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
        )
    return summary


if __name__ == '__main__':
    print(json.dumps(check(write_report=True), ensure_ascii=False, indent=2))
