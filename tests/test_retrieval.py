import numpy as np
import pytest
import torch

from certainrag.signals.retrieval_score import RetrievalSignal
from certainrag.utils import pack_windows, split_sentences

QUESTION = "what is the phone number of the shop in hayward ?"
LONG_CHUNK = " ".join(["the shop in hayward has a phone number ."] * 80)
CHUNKS = ["the shop is on hayward street .", LONG_CHUNK, "", "weather in spring ."]


@pytest.fixture(scope="module")
def signal(tiny_reranker):
    return RetrievalSignal(tiny_reranker, device="cpu")


def direct_logits(signal, windows):
    batch = signal.tokenizer([QUESTION] * len(windows), windows, padding=True, return_tensors="pt")
    with torch.no_grad():
        return signal.model(**batch).logits[:, 0].numpy()


def test_long_chunk_scored_as_max_over_windows(signal):
    budget = signal.max_length - len(signal.tokenizer(QUESTION, add_special_tokens=False)["input_ids"]) - 3
    windows = pack_windows(split_sentences(LONG_CHUNK), signal.tokenizer, budget)
    assert len(windows) > 1
    assert max(len(signal.tokenizer(QUESTION, w)["input_ids"]) for w in windows) <= signal.max_length
    expected = [
        direct_logits(signal, [CHUNKS[0]])[0],
        direct_logits(signal, windows).max(),
        direct_logits(signal, [""])[0],
        direct_logits(signal, [CHUNKS[3]])[0],
    ]
    np.testing.assert_allclose(signal.logits(QUESTION, CHUNKS), expected, atol=1e-5)


def test_batch_size_does_not_change_scores(tiny_reranker, signal):
    small = RetrievalSignal(tiny_reranker, device="cpu", batch_size=2)
    np.testing.assert_allclose(small.logits(QUESTION, CHUNKS), signal.logits(QUESTION, CHUNKS), atol=1e-5)


def test_score_applies_platt_and_takes_max(tiny_reranker):
    signal = RetrievalSignal(tiny_reranker, device="cpu", platt=(2.0, -1.0))
    logits = signal.logits(QUESTION, CHUNKS)
    result = signal.score(QUESTION, CHUNKS)
    np.testing.assert_allclose(result.chunk_scores, 1 / (1 + np.exp(-(2.0 * logits - 1.0))), atol=1e-6)
    assert result.score == max(result.chunk_scores)


def test_no_chunks(signal):
    result = signal.score(QUESTION, [])
    assert result.score == 0.0
    assert result.chunk_scores == []
