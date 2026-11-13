"""Restore a cut paragraph through an independently cited, unchanged neighbor chunk."""
import re


def body(document):
    text = document.content
    title = document.title.strip()
    while title and text.startswith(title + '\n'):
        text = text[len(title)+1:]
    return ' '.join(text.split())


def expand_adjacent(documents, hits, max_extra=1):
    positions = {d.id:i for i,d in enumerate(documents)}
    selected = {d.id for d in hits}
    result = list(hits)
    for hit in hits:
        position = positions[hit.id]
        if len(result)-len(hits) >= max_extra:
            break
        if position+1 >= len(documents):
            continue
        following = documents[position+1]
        if following.id in selected:
            continue
        if (hit.metadata.get('document_id') != following.metadata.get('document_id')
                or hit.metadata.get('warehouse') != following.metadata.get('warehouse')
                or hit.metadata.get('section_id') != following.metadata.get('section_id')):
            continue
        left, right = body(hit), body(following)
        # An exact substantial suffix/prefix overlap is evidence of a continuation.
        overlap = next((n for n in range(min(len(left),len(right),600),59,-1) if left[-n:]==right[:n]), 0)
        if not overlap or re.search(r'[.!?。]\s*$', left):
            continue
        neighbor = following.model_copy(deep=True)
        neighbor.metadata['retrieval'] = {'method':'adjacent_context_v1','parent_hit_id':hit.id,
                                          'overlap_characters':overlap}
        result.append(neighbor)
        selected.add(neighbor.id)
    return result
