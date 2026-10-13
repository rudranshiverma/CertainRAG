from functools import lru_cache

import pysbd


@lru_cache(maxsize=1)
def _segmenter():
    return pysbd.Segmenter(language="en", clean=False)


def split_sentences(text):
    sentences = []
    for line in text.splitlines():
        if line.strip():
            sentences.extend(s.strip() for s in _segmenter().segment(line) if s.strip())
    return sentences


def pack_windows(sentences, tokenizer, max_tokens):
    if not sentences:
        return []
    units = []
    for sentence, ids in zip(sentences, tokenizer(sentences, add_special_tokens=False)["input_ids"]):
        if len(ids) <= max_tokens:
            units.append((sentence, len(ids)))
            continue
        for i in range(0, len(ids), max_tokens):
            piece = ids[i : i + max_tokens]
            units.append((tokenizer.decode(piece), len(piece)))
    windows, current, size = [], [], 0
    for text, n in units:
        if current and size + n > max_tokens:
            windows.append(" ".join(current))
            current, size = [], 0
        current.append(text)
        size += n
    if current:
        windows.append(" ".join(current))
    return windows
