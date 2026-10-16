import re

from certainrag import CertainRAG, Guard

DOCUMENTS = [
    "Metformin is the first-line medication for type 2 diabetes. It lowers blood glucose mainly by reducing "
    "glucose production in the liver.",
    "Common side effects of metformin include nausea, diarrhoea and abdominal discomfort. Lactic acidosis is a rare "
    "but serious complication.",
    "The Great Barrier Reef is the world's largest coral reef system, located off the coast of Queensland, Australia.",
]


def words(text):
    return set(re.findall(r"\w+", text.lower()))


def retrieve(question, k=2):
    return sorted(DOCUMENTS, key=lambda d: len(words(question) & words(d)), reverse=True)[:k]


def generate(question, chunks, strict):
    # extractive stand-in for an LLM call: returns the sentence that overlaps most with the question
    sentences = [s.strip().rstrip(".") + "." for chunk in chunks for s in chunk.split(". ")]
    return max(sentences, key=lambda s: len(words(question) & words(s)))


def main():
    rag = CertainRAG()

    question = "What are the common side effects of metformin?"
    chunks = retrieve(question)
    for answer in [
        "Common side effects of metformin include nausea, diarrhoea and abdominal discomfort.",
        "Metformin commonly causes weight gain and hair loss.",
    ]:
        result = rag.score(question, answer, chunks)
        print(f"\nQ: {question}\nA: {answer}")
        print(f"decision={result.decision.value} retrieval={result.retrieval.score:.2f}")
        if result.faithfulness is not None:
            print(f"faithfulness={result.faithfulness.score:.2f}")
            for claim in result.faithfulness.claims:
                print(f"  {claim.score:.2f}  {claim.text}")

    guard = Guard(rag, retrieve, generate)
    for question in ["How does metformin lower blood glucose?", "Who won the 2018 football World Cup?"]:
        out = guard.run(question)
        print(f"\nQ: {question}\nA: {out.answer}")
        print(f"decision={out.result.decision.value} steps={out.steps}")


if __name__ == "__main__":
    main()
