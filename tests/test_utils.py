from certainrag.utils import pack_windows, split_sentences


def token_count(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def test_split_keeps_abbreviations_together():
    text = "The study (e.g. Smith et al.) found a 3.5% rise in U.S. cases. Dr. Lee agreed!"
    assert split_sentences(text) == [
        "The study (e.g. Smith et al.) found a 3.5% rise in U.S. cases.",
        "Dr. Lee agreed!",
    ]


def test_split_treats_lines_as_boundaries():
    assert split_sentences("- item one\n\n- item two\n") == ["- item one", "- item two"]


def test_split_empty_input():
    assert split_sentences("") == []
    assert split_sentences("  \n \n") == []


def test_pack_respects_budget_and_keeps_text(tokenizer):
    sentences = ["the trial ran two years in boston ."] * 9 + ["aspirin reduced mortality ."]
    windows = pack_windows(sentences, tokenizer, max_tokens=20)
    assert len(windows) > 1
    assert all(token_count(tokenizer, w) <= 20 for w in windows)
    assert " ".join(windows).split() == " ".join(sentences).split()


def test_pack_splits_oversized_sentence(tokenizer):
    windows = pack_windows(["the " * 100], tokenizer, max_tokens=30)
    assert [token_count(tokenizer, w) for w in windows] == [30, 30, 30, 10]


def test_pack_empty():
    assert pack_windows([], None, 10) == []
