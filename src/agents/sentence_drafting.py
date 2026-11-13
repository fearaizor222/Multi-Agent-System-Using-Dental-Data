"""Opt-in single-call A2 for an adapter trained on the sentence selection contract."""
import json
from src.training.a2_sentence_protocol import make_messages, normalize_payload_evidence, resolve_selection


def draft_with_sentences(generate, payload):
    """generate(messages) -> raw JSON string; invalid output raises, no silent fallback.

    Caller must still apply semantic review (A3). Correct IDs do not prove that
    the selected source answers the question or supports the paraphrase.
    """
    payload = normalize_payload_evidence(payload)
    raw = generate(make_messages(payload))
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate output key')
            result[key] = value
        return result
    selected = json.loads(raw, object_pairs_hook=unique)
    return resolve_selection(selected, payload)
