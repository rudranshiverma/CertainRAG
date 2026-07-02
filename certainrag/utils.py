import math
from .exceptions import MissingDependencyError

DEFAULT_EMBEDDER_MODEL="sentence-transformers/all-MiniLM-L6-v2"

def get_embedder(model_name=DEFAULT_EMBEDDER_MODEL):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise MissingDependencyError(
            "sentence-transformers is required for the self-consistency signal. "
            "Install with: pip install sentence-transformers"
        ) from e
    return SentenceTransformer(model_name)

def split_into_sentences(text):
    text=text.strip()
    if not text:
        return []
    sentences=[]
    current=""
    for char in text:
        current+=char
        if char in ".!?":
            sentences.append(current.strip())
            current=""
    if current.strip():
        sentences.append(current.strip())
    return sentences

def rank_weights(n):
    if n<=0:
        return []
    raw_weights=[1.0/math.log2(i+2) for i in range(n)]
    total=sum(raw_weights)
    return [w/total for w in raw_weights]

def weighted_mean(values, weights):
    if not values:
        return None
    total=0.0
    for value, weight in zip(values, weights):
        total+=value*weight
    return total