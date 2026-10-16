import pytest
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    DebertaV2Config,
    DebertaV2ForSequenceClassification,
    PreTrainedTokenizerFast,
)

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
WORDS = (
    "the a an of in on to and is was by for with at from it its that this these were has had be are not no yes "
    "aspirin reduced mortality trial ran two years boston paris patients study found rise cases dr lee agreed "
    "shop phone number hayward street report company grew revenue rose sharply weather spring item one "
    "what is which who when where how why did does question answer context "
    "1 2 3 5 12 2015 % . , ? ! ( ) - : e g u s al et fig"
).split()


@pytest.fixture(scope="session")
def tokenizer():
    vocab = {token: i for i, token in enumerate(SPECIAL + list(dict.fromkeys(WORDS)))}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        eos_token="[SEP]",
        model_max_length=512,
    )


@pytest.fixture(scope="session")
def tiny_reranker(tokenizer, tmp_path_factory):
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        num_labels=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    path = tmp_path_factory.mktemp("reranker")
    BertForSequenceClassification(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tiny_verifier(tokenizer, tmp_path_factory):
    torch.manual_seed(0)
    config = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id,
        relative_attention=True,
        position_buckets=64,
        pos_att_type=["p2c", "c2p"],
        position_biased_input=False,
    )
    path = tmp_path_factory.mktemp("verifier")
    DebertaV2ForSequenceClassification(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


class FakeVerifier:
    """Supported when every claim word appears in the window."""

    def __init__(self, tokenizer, budget=20):
        self.tokenizer = tokenizer
        self.budget = budget
        self.calls = 0

    def prepare_claims(self, claims):
        return claims

    def window_budget(self, claims):
        return self.budget

    def score(self, windows, claims):
        self.calls += 1
        out = []
        for window, claim in zip(windows, claims, strict=True):
            words = set(window.lower().replace(".", " ").split())
            claim_words = claim.lower().replace(".", " ").split()
            out.append(0.9 if all(w in words for w in claim_words) else 0.1)
        return out


class ConstantVerifier(FakeVerifier):
    def __init__(self, tokenizer, values):
        super().__init__(tokenizer)
        self.values = list(values)

    def score(self, windows, claims):
        self.calls += 1
        value = self.values.pop(0) if len(self.values) > 1 else self.values[0]
        return [value] * len(windows)


class FakeRetrieval:
    def __init__(self, score):
        self.score_value = score

    def score(self, question, chunks):
        from certainrag import RetrievalResult

        return RetrievalResult(score=self.score_value, chunk_scores=[self.score_value] * len(chunks))


@pytest.fixture
def fake_verifier(tokenizer):
    return FakeVerifier(tokenizer)


@pytest.fixture
def constant_verifier(tokenizer):
    return lambda *values: ConstantVerifier(tokenizer, values)


@pytest.fixture
def fake_retrieval():
    return FakeRetrieval
