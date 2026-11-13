"""Build Stage 2 (v4) cleaned, shortcut-free A2 sentence selection dataset.

Outputs to `data/processed/a2_sentence_selection_v4` without modifying frozen
`a2_sentence_selection_v2_reviewed` or `a2_sentence_selection_v3_hard` baselines.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile

import scripts.build_a2_sentence_v2 as v2b
from scripts.a2_v4_cleaning import clean_hard_source, clean_seed_record, clean_source_text
from scripts.a2_v4_specs import v4_challenge_specs, v4_train_specs
from src.training.a2_sentence_protocol import (
    SentenceDraft,
    make_messages,
    resolve_selection,
    selection_from_gold,
    source_units,
)

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / 'data/processed/a2_raft_pilot_v2/seed_review.json'
HARD_SOURCES_PATH = ROOT / 'data/curated/a2_hard_v3_sources.json'
OUTPUT = ROOT / 'data/processed/a2_sentence_selection_v4'

PROPER_STARTERS = {
    'Ash', 'American', 'Belmont', 'Bennett', 'Box', 'Bảng', 'Candida', 'CHIP',
    'DMFT', 'EOB', 'FDA', 'Finlay', 'Frankfort', 'Hình', 'HMO', 'Hoàng', 'IRB',
    'KCTT', 'LMTĐ', 'Medicaid', 'Nguyễn', 'Nuremberg', 'OHI', 'OHI-S', 'Palla',
    'PCR', 'PLI', 'Posselt', 'Quick-Master', 'Ramfjord', 'Salaorni', 'Spee',
    'TRICARE', 'TQTT', 'Tell-show-do', 'Tuskegee', 'Type', 'UCR', 'VAS', 'Visual',
}

CONNECTORS = (
    ('ngoai_ra', 'Ngoài ra,'),
    ('dong_thoi', 'Đồng thời,'),
    ('ben_canh_do', 'Bên cạnh đó,'),
    ('them_vao_do', 'Thêm vào đó,'),
    ('va_compound', ', và'),
)

EV_SCHEDULE = (1, 2, 3, 2, 1, 2, 3, 2, 1, 3)

# Topic-disjoint cluster tags for safe intra-split distractor selection on train.
TRAIN_TOPIC_CLUSTER: dict[int, str] = {
    **{p: 'tmj_muscles' for p in (520, 521, 522, 523, 524, 525, 526, 653, 655)},
    **{p: 'rhm_occlusion' for p in (555, 561, 607, 608, 610, 612, 627, 628)},
    **{p: 'rhm_pain_history' for p in (643, 644, 646)},
    **{p: 'dph_financing' for p in (151, 152, 153)},
    **{p: 'dph_evaluation' for p in (262, 263, 268, 269)},
    **{p: 'dph_ethics' for p in (303, 305, 308)},
    **{p: 'dph_biostats_design' for p in (322, 325, 326, 328, 357, 359, 360, 363)},
    **{p: 'dph_epi_lesions' for p in (404, 478)},
}


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decap_clause(text: str) -> str:
    if not text:
        return text
    first_word = text.split(maxsplit=1)[0].rstrip('.,:;?!()')
    if first_word in PROPER_STARTERS:
        return text
    if len(first_word) >= 2 and first_word[0].isupper() and first_word[1].isupper():
        return text
    return text[0].lower() + text[1:]


def join_questions(q1: str, q2: str, connector_index: int) -> tuple[str, str]:
    """Join two questions naturally with a rotated Vietnamese connector."""
    tag, token = CONNECTORS[connector_index % len(CONNECTORS)]
    c2 = _decap_clause(q2.strip())
    if token.startswith(','):
        base = q1.strip().rstrip('?').rstrip()
        return f'{base}{token} {c2}', tag
    return f'{q1.strip()} {token} {c2}', tag


def _build_record(
    *,
    sample_id: str,
    split: str,
    payload: dict,
    gold: dict,
    metadata: dict,
) -> dict:
    selected = selection_from_gold(gold, payload)
    target = selected.model_dump()
    SentenceDraft.model_validate(target)
    messages = make_messages(payload) + [
        {'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False)}
    ]
    return {
        'record': {'messages': messages},
        'metadata': dict(metadata, id=sample_id, split=split, status=gold['status']),
        'evidence_count': len(payload['evidence']),
    }


def _reconstruct_single_owner_evidence(seed: dict, kept: list[int], retained_spans_list: list[dict]) -> list[dict]:
    """Reconstruct owner-only evidence strictly from provenance spans on cleaned seed."""
    items = []
    for prov in retained_spans_list:
        sk = prov['source_key']
        content = seed['sources'][sk]['content']
        units = source_units(content)
        kept_units = [u['text'] for u in units if [u['start'], u['end']] in prov['retained_spans']]
        if not kept_units:
            # Fallback using v2b removal policy directly on cleaned source.
            raise ValueError(f'Empty retained spans on {seed["id"]}/{sk}')
        items.append({'evidence_id': 'I1', 'title': seed['sources'][sk]['title'], 'content': ' '.join(kept_units)})
    return items


def _build_cleaned_v2_splits(seeds_clean: list[dict]) -> tuple[list[dict], list[dict]]:
    """Replay v2 construction on cleaned seeds, downsample keep=[], and add 1-ev diversity."""
    seed_by_id = {s['id']: s for s in seeds_clean}
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        clean_seed_path = td_path / 'seed_review.json'
        clean_seed_path.write_text(json.dumps(seeds_clean, ensure_ascii=False), encoding='utf-8')
        cand_dir = td_path / 'candidates'

        orig_source = v2b.SOURCE
        v2b.SOURCE = clean_seed_path
        try:
            v2b.build(output=cand_dir)
        finally:
            v2b.SOURCE = orig_source

        raw_cands = [
            json.loads(line)
            for line in (cand_dir / 'candidates.jsonl').read_text(encoding='utf-8').splitlines()
        ]

    val_rows: list[dict] = []
    train_draft_or_hard: list[dict] = []
    train_keep_empty: dict[str, list[dict]] = defaultdict(list)

    for cand in raw_cands:
        seed = seed_by_id[cand['family_id']]
        assistant = json.loads(cand['messages'][-1]['content'])
        # Ensure assistant JSON follows SelectedClaim field order + patterns.
        validated = SentenceDraft.model_validate(assistant).model_dump()
        messages = cand['messages'][:2] + [
            {'role': 'assistant', 'content': json.dumps(validated, ensure_ascii=False)}
        ]
        docs = sorted({v['document_id'] for v in seed['sources'].values()})
        meta = {
            'id': cand['id'],
            'split': cand['split'],
            'kind': 'base_reviewed_clean',
            'variant': 'v2_rev_clean',
            'family_id': cand['family_id'],
            'language': cand['language'],
            'document_id': docs[0] if len(docs) == 1 else ','.join(docs),
            'document_ids': docs,
            'evidence_count': len(cand['original_evidence']),
            'kept_fact_indices': cand['kept_fact_indices'],
            'question_fact_indices': cand['question_fact_indices'],
            'status': cand['proposed_status'],
            'distractor': cand['distractor'],
            'source_provenance': cand['source_provenance'],
            'source_review': clean_source_text(seed['review']),
        }
        entry = {'record': {'messages': messages}, 'metadata': meta, '_cand': cand}
        if cand['split'] == 'val':
            val_rows.append(entry)
        elif cand['proposed_status'] == 'insufficient_evidence' and len(cand['kept_fact_indices']) == 0:
            train_keep_empty[cand['family_id']].append(entry)
        else:
            train_draft_or_hard.append(entry)

    # Downsample weak keep=[] from 144 -> 46 (2 per family across all 23 keep=[] families).
    kept_weak: list[dict] = []
    for fam in sorted(train_keep_empty):
        group = sorted(train_keep_empty[fam], key=lambda item: item['metadata']['id'])
        if len(group) <= 2:
            kept_weak.extend(group)
        else:
            # Prefer one non-combined question variant and one distinct variant.
            kept_weak.extend([group[0], group[len(group) // 2]])

    v2_train_all = train_draft_or_hard + kept_weak
    v2_train_all.sort(key=lambda item: item['metadata']['id'])

    # Convert ~25% of single-owner v2_rev train rows to 1-evidence (drop distractor).
    seen_prompts: set[str] = set()
    final_v2_train: list[dict] = []
    # Reserve existing multi-evidence prompts first for candidates that stay multi-ev.
    single_ev_eligible = [
        idx for idx, item in enumerate(v2_train_all)
        if len(item['_cand']['source_provenance']) == 1 and idx % 4 == 0
    ]
    convert_set = set(single_ev_eligible)

    # Build converted entries first, verify prompt uniqueness, fall back if duplicate prompt.
    for idx, item in enumerate(v2_train_all):
        cand = item['_cand']
        meta = item['metadata']
        converted = None
        if idx in convert_set:
            seed = seed_by_id[cand['family_id']]
            try:
                single_ev = _reconstruct_single_owner_evidence(
                    seed, cand['kept_fact_indices'], cand['source_provenance']
                )
                u_payload = json.loads(cand['messages'][1]['content'].split('\nJSON schema:\n')[0])
                question = u_payload['question']
                payload = {
                    'question': question,
                    'evidence': single_ev,
                    'allowed_evidence_ids': ['I1'],
                }
                q_idx = cand['question_fact_indices']
                kept = set(cand['kept_fact_indices'])
                supported = [i for i in q_idx if i in kept]
                claims = []
                for i in supported:
                    f = seed['claims'][i]
                    claims.append({
                        'text': f['text'],
                        'evidence_ids': ['I1'],
                        'supporting_quotes': [{'evidence_id': 'I1', 'quote': f['quote']}],
                    })
                gold = {'status': 'draft' if claims else 'insufficient_evidence', 'claims': claims}
                built = _build_record(
                    sample_id=f"{meta['id']}_ev1",
                    split='train',
                    payload=payload,
                    gold=gold,
                    metadata=dict(
                        meta,
                        evidence_count=1,
                        variant='v2_rev_clean_single_ev',
                        distractor=None,
                    ),
                )
                prompt_key = json.dumps(built['record']['messages'][:2], ensure_ascii=False, sort_keys=True)
                if prompt_key not in seen_prompts:
                    seen_prompts.add(prompt_key)
                    converted = {'record': built['record'], 'metadata': built['metadata']}
            except ValueError:
                converted = None

        if converted is not None:
            final_v2_train.append(converted)
        else:
            prompt_key = json.dumps(item['record']['messages'][:2], ensure_ascii=False, sort_keys=True)
            seen_prompts.add(prompt_key)
            final_v2_train.append({'record': item['record'], 'metadata': meta})

    val_clean = [{'record': item['record'], 'metadata': item['metadata']} for item in val_rows]
    return final_v2_train, val_clean


def _pick_distractors(
    pidx: int,
    split_sources: dict[int, dict],
    n_dist: int,
    rank: int,
    task_idx: int,
    split: str,
) -> list[int]:
    if n_dist <= 0:
        return []
    owner_doc = split_sources[pidx]['document_id']
    if split == 'challenge':
        candidates = sorted(k for k, v in split_sources.items() if v['document_id'] != owner_doc)
        start = (rank + task_idx) % len(candidates)
        rotated = candidates[start:] + candidates[:start]
        return rotated[:n_dist]

    owner_cluster = TRAIN_TOPIC_CLUSTER[pidx]
    is_rhm = owner_doc.startswith('RHM1-')
    same_lang_safe = sorted(
        k for k, v in split_sources.items()
        if k != pidx
        and v['document_id'] != owner_doc
        and TRAIN_TOPIC_CLUSTER[k] != owner_cluster
        and v['document_id'].startswith('RHM1-') == is_rhm
        and not (
            {owner_cluster, TRAIN_TOPIC_CLUSTER[k]}
            <= {'tmj_muscles', 'rhm_occlusion'}
        )
        and not (
            {owner_cluster, TRAIN_TOPIC_CLUSTER[k]}
            <= {'dph_biostats_design', 'dph_evaluation', 'dph_ethics'}
        )
    )
    cross_lang = sorted(
        k for k, v in split_sources.items()
        if v['document_id'].startswith('RHM1-') != is_rhm
        and v['document_id'] != owner_doc
    )

    if n_dist == 1:
        use_same_lang = ((rank + task_idx) % 2 == 0) and bool(same_lang_safe)
        pool = same_lang_safe if use_same_lang else cross_lang
        return [pool[(rank * 3 + task_idx) % len(pool)]]

    # For n_dist == 2: pair 1 safe same-language distractor + 1 cross-language distractor when available.
    first = (
        same_lang_safe[(rank + task_idx) % len(same_lang_safe)]
        if same_lang_safe
        else cross_lang[(rank + task_idx) % len(cross_lang)]
    )
    second_pool = [k for k in cross_lang if k != first]
    if not second_pool:
        second_pool = [k for k in same_lang_safe if k != first]
    second = second_pool[(rank * 2 + task_idx) % len(second_pool)]
    return [first, second]


def _make_intact_family(
    *,
    sp: dict,
    rank: int,
    split: str,
    split_sources: dict[int, dict],
) -> list[dict]:
    pidx = sp['pool_index']
    src = split_sources[pidx]
    facts = sp['facts']
    lang = 'vi' if src['document_id'].startswith('RHM1-') else 'en'
    neg_base = {'question': sp['negative_question'], 'review': sp['negative_review']}
    c0, c1 = sp['minimal_contrast_negatives']

    # Build 10 balanced tasks per chunk.
    q_two_01, conn_01 = join_questions(facts[0]['question'], facts[1]['question'], rank + 0)
    q_two_12, conn_12 = join_questions(facts[1]['question'], facts[2]['question'], rank + 2)

    if rank % 2 == 0:
        partial_pos_neg_q, conn_pn = join_questions(facts[2]['question'], neg_base['question'], rank + 1)
        partial_pos_neg_rev = neg_base['review']
        partial_neg_pos_q, conn_np = join_questions(c0['question'], facts[0]['question'], rank + 3)
        partial_neg_pos_rev = c0['review']
    else:
        partial_pos_neg_q, conn_pn = join_questions(facts[2]['question'], c1['question'], rank + 4)
        partial_pos_neg_rev = c1['review']
        partial_neg_pos_q, conn_np = join_questions(neg_base['question'], facts[0]['question'], rank + 1)
        partial_neg_pos_rev = neg_base['review']

    tasks = [
        ('single_aspect', facts[0]['question'], [0], 'single', 'none', ''),
        ('single_aspect', facts[1]['question'], [1], 'single', 'none', ''),
        ('single_aspect', facts[2]['question'], [2], 'single', 'none', ''),
        ('two_supported_aspects', q_two_01, [0, 1], conn_01, 'none', ''),
        ('two_supported_aspects', q_two_12, [1, 2], conn_12, 'none', ''),
        ('intact_same_entity_missing_aspect', neg_base['question'], [], 'none', 'none', neg_base['review']),
        ('minimal_contrast_negative', c0['question'], [], 'none', 'none', c0['review']),
        ('minimal_contrast_negative', c1['question'], [], 'none', 'none', c1['review']),
        ('intact_partial_support', partial_pos_neg_q, [2], conn_pn, 'pos_first', partial_pos_neg_rev),
        ('intact_partial_support', partial_neg_pos_q, [0], conn_np, 'neg_first', partial_neg_pos_rev),
    ]

    built_items: list[dict] = []
    for task_idx, (kind, question, supported, conn_tag, partial_dir, neg_review) in enumerate(tasks):
        ev_count = EV_SCHEDULE[(rank + task_idx) % len(EV_SCHEDULE)]
        distractor_ids = _pick_distractors(pidx, split_sources, ev_count - 1, rank, task_idx, split)
        ordered_sources: list[tuple[int, dict]] = [(pidx, src)] + [
            (did, split_sources[did]) for did in distractor_ids
        ]
        # Deterministic position shift so primary passage is not always I1.
        pos = (rank + task_idx * 2) % ev_count
        rotated = ordered_sources[-pos:] + ordered_sources[:-pos] if pos else ordered_sources
        owner_eid = f'I{rotated.index((pidx, src)) + 1}'

        evidence = [
            {'evidence_id': f'I{i}', 'title': s['title'], 'content': s['content']}
            for i, (_, s) in enumerate(rotated, 1)
        ]
        payload = {
            'question': question,
            'evidence': evidence,
            'allowed_evidence_ids': [e['evidence_id'] for e in evidence],
        }
        claims = [
            {
                'text': facts[i]['text'],
                'evidence_ids': [owner_eid],
                'supporting_quotes': [{'evidence_id': owner_eid, 'quote': facts[i]['needle']}],
            }
            for i in supported
        ]
        gold = {'status': 'draft' if claims else 'insufficient_evidence', 'claims': claims}
        sample_id = f'v4_{split}_{src["chunk_id"][:14]}_{pidx}_{task_idx}'
        meta = {
            'kind': kind,
            'variant': f'v4_intact_{split}',
            'family_id': src['chunk_id'],
            'pool_index': pidx,
            'chunk_id': src['chunk_id'],
            'document_id': src['document_id'],
            'language': lang,
            'parent_id': src['parent_id'],
            'evidence_count': ev_count,
            'owner_evidence_id': owner_eid,
            'distractor_pool_indices': distractor_ids,
            'distractor_documents': [split_sources[d]['document_id'] for d in distractor_ids],
            'supported_fact_indices': supported,
            'connector': conn_tag,
            'partial_direction': partial_dir,
            'negative_review': neg_review,
            'source_intact': True,
            'source_review': (
                f'cleaned v4 source + audited 2-contrast negatives ({split}); not clinical expert review'
            ),
        }
        built = _build_record(
            sample_id=sample_id,
            split=split,
            payload=payload,
            gold=gold,
            metadata=meta,
        )
        # Validate resolution before returning.
        resolve_selection(json.loads(built['record']['messages'][-1]['content']), payload)
        built_items.append({'record': built['record'], 'metadata': built['metadata']})
    return built_items


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def build(output: Path = OUTPUT, overwrite: bool = False) -> dict:
    if output.exists() and not overwrite:
        raise ValueError(f'Refusing to overwrite existing {output}; pass overwrite=True if intended.')

    seeds_raw = json.loads(SEED_PATH.read_text(encoding='utf-8'))
    seeds_clean = [clean_seed_record(s) for s in seeds_raw]

    hard_raw = json.loads(HARD_SOURCES_PATH.read_text(encoding='utf-8'))
    hard_clean = {int(k): clean_hard_source(v) for k, v in hard_raw.items()}

    v2_train, val_rows = _build_cleaned_v2_splits(seeds_clean)

    t_specs = v4_train_specs()
    c_specs = v4_challenge_specs()
    train_sources = {sp['pool_index']: hard_clean[sp['pool_index']] for sp in t_specs}
    challenge_sources = {sp['pool_index']: hard_clean[sp['pool_index']] for sp in c_specs}

    v4_intact_train: list[dict] = []
    for rank, sp in enumerate(t_specs):
        v4_intact_train.extend(
            _make_intact_family(sp=sp, rank=rank, split='train', split_sources=train_sources)
        )

    challenge_rows: list[dict] = []
    for rank, sp in enumerate(c_specs):
        challenge_rows.extend(
            _make_intact_family(sp=sp, rank=rank, split='challenge', split_sources=challenge_sources)
        )

    all_train = v2_train + v4_intact_train
    random.Random(2026).shuffle(all_train)

    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / 'train.jsonl', [r['record'] for r in all_train])
    _write_jsonl(output / 'train.metadata.jsonl', [r['metadata'] for r in all_train])
    _write_jsonl(output / 'val.jsonl', [r['record'] for r in val_rows])
    _write_jsonl(output / 'val.metadata.jsonl', [r['metadata'] for r in val_rows])
    _write_jsonl(output / 'challenge.jsonl', [r['record'] for r in challenge_rows])
    _write_jsonl(output / 'challenge.metadata.jsonl', [r['metadata'] for r in challenge_rows])

    source_review = {
        'version': 'a2_sentence_selection_v4',
        'val_families': sorted({r['metadata']['family_id'] for r in val_rows}),
        'train_intact_pool_indices': [sp['pool_index'] for sp in t_specs],
        'challenge_pool_indices': [sp['pool_index'] for sp in c_specs],
        'train_documents': sorted(
            {d for r in all_train for d in ([r['metadata']['document_id']] if isinstance(r['metadata']['document_id'], str) and ',' not in r['metadata']['document_id'] else r['metadata'].get('document_ids', []))}
        ),
        'val_documents': sorted(
            {d for r in val_rows for d in r['metadata']['document_ids']}
        ),
        'challenge_documents': sorted({r['metadata']['document_id'] for r in challenge_rows}),
        'cleaning_protocol': 'normalize_source_content + scripts.a2_v4_cleaning.OCR_REPLACEMENTS',
    }
    (output / 'source_review.json').write_text(
        json.dumps(source_review, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    manifest = {
        'version': 'a2_sentence_selection_v4',
        'counts': {
            'train': len(all_train),
            'train_v2_rev_clean': len(v2_train),
            'train_v4_intact': len(v4_intact_train),
            'val': len(val_rows),
            'challenge': len(challenge_rows),
        },
        'statuses': {
            'train': dict(Counter(r['metadata']['status'] for r in all_train)),
            'val': dict(Counter(r['metadata']['status'] for r in val_rows)),
            'challenge': dict(Counter(r['metadata']['status'] for r in challenge_rows)),
        },
        'evidence_counts': {
            'train': dict(Counter(r['metadata']['evidence_count'] for r in all_train)),
            'val': dict(Counter(r['metadata']['evidence_count'] for r in val_rows)),
            'challenge': dict(Counter(r['metadata']['evidence_count'] for r in challenge_rows)),
        },
        'protocol_sha256': sha256_bytes(ROOT / 'src/training/a2_sentence_protocol.py'),
        'cleaning_sha256': sha256_bytes(ROOT / 'scripts/a2_v4_cleaning.py'),
        'specs_sha256': sha256_bytes(ROOT / 'scripts/a2_v4_specs.py'),
        'val_sha256': sha256_bytes(output / 'val.jsonl'),
        'challenge_sha256': sha256_bytes(output / 'challenge.jsonl'),
        'test_examples_used': False,
        'preserved_v3_baseline': 'data/processed/a2_sentence_selection_v3_hard',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    print(json.dumps(build(overwrite=True), ensure_ascii=False, indent=2))
