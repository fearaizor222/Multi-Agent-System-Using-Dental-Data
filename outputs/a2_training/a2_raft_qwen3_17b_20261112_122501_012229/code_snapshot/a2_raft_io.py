"""Read the validated pilot and score source quotes separately from claim semantics."""
import json
from pathlib import Path

from src.training.a2_raft_protocol import GroundedDraft


def validate_quotes(draft, payload):
    sources = {item['evidence_id']:item['content'] for item in payload['evidence']}
    for claim in draft.claims:
        for quote in claim.supporting_quotes:
            if quote.evidence_id not in sources or quote.quote not in sources[quote.evidence_id]:
                raise ValueError('Quote is not verbatim in the cited source')


def score_quotes(predicted, gold, payload):
    """Reference coverage is a lexical check, not evidence of semantic correctness."""
    try:
        validate_quotes(predicted,payload)
        verbatim = True
    except ValueError:
        verbatim = False
    expected = [(q['evidence_id'],q['quote']) for c in gold['claims'] for q in c['supporting_quotes']]
    actual = [(q.evidence_id,q.quote) for c in predicted.claims for q in c.supporting_quotes]
    covered = sum(any(eid==actual_id and quote in text for actual_id,text in actual) for eid,quote in expected)
    return {'quotes_verbatim':verbatim,'reference_quotes_count':len(expected),
        'reference_quotes_covered':covered,
        'reference_proof_coverage':covered/len(expected) if expected else None,
        'reference_evidence_match': {i for c in predicted.claims for i in c.evidence_ids} ==
                                    {i for c in gold['claims'] for i in c['evidence_ids']}}


def load_raft_data(directory):
    # Full provenance checks run locally before a packet is sent to any GPU job.
    from scripts.check_a2_raft_pilot import check
    directory = Path(directory)
    verification = check(directory,write_reports=False)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    packet = {}
    for source_split, name in (('train','train'),('val','validation')):
        records = [json.loads(x) for x in (directory/(source_split+'.jsonl')).read_text(encoding='utf-8').splitlines() if x.strip()]
        metas = [json.loads(x) for x in (directory/(source_split+'.metadata.jsonl')).read_text(encoding='utf-8').splitlines() if x.strip()]
        packet[name] = [{**row,'metadata':meta} for row,meta in zip(records,metas)]
    packet['info'] = {'dataset_version':manifest['version'],'output_contract':'a2_grounded_quotes_v1',
        'source_sha256':manifest['files_sha256'],'counts':{k:len(packet[k]) for k in packet},
        'split_method':manifest['split_policy'],'test_reserved':True,'scope':manifest['scope'],
        'verification':verification}
    return packet
