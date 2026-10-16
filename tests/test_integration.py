import pytest

from certainrag import CertainRAG, Decision, Policy

pytestmark = pytest.mark.slow

CONTEXT = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was completed in 1889 for the World's Fair "
    "and is 330 metres tall.",
    "Mount Everest is the highest mountain above sea level, located in the Himalayas on the border of Nepal and China.",
]
QUESTION = "When was the Eiffel Tower completed and how tall is it?"
SUPPORTED = "The Eiffel Tower was completed in 1889 and is 330 metres tall."
CONTRADICTED = "The Eiffel Tower was completed in 1925 and is 120 metres tall."
UNRELATED_QUESTION = "What is the boiling point of mercury?"


@pytest.fixture(scope="module")
def rag():
    return CertainRAG(policy=Policy(retrieval_threshold=0.0, faithfulness_threshold=0.0))


def test_relevant_chunk_ranks_first(rag):
    result = rag.score_retrieval(QUESTION, CONTEXT)
    assert result.chunk_scores[0] > result.chunk_scores[1]


def test_unrelated_question_scores_lower(rag):
    related = rag.score_retrieval(QUESTION, CONTEXT).score
    unrelated = rag.score_retrieval(UNRELATED_QUESTION, CONTEXT).score
    assert unrelated < related


def test_supported_answer_beats_contradicted(rag):
    supported = rag.score_faithfulness(QUESTION, SUPPORTED, CONTEXT)
    contradicted = rag.score_faithfulness(QUESTION, CONTRADICTED, CONTEXT)
    assert supported.score > 0.5 > contradicted.score


def test_end_to_end_result(rag):
    result = rag.score(QUESTION, SUPPORTED, CONTEXT)
    assert result.decision == Decision.ACCEPT
    assert result.faithfulness.method == "verifier"
    assert len(result.retrieval.chunk_scores) == len(CONTEXT)
