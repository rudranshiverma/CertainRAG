#Integration test simulates a real developer using CertainRAG as a library on top of their own RAG pipeline.
#This test uses a real PDF, real retrieval, and a real LLM.
import pytest
from certainrag import CertainRAG
from certainrag.utils import normalize_l2

pytest.importorskip("langchain_ollama")

def get_test_pipeline():
    #a minimal RAG pipeline for testing
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import ChatOllama
    import os

    pdf_path="tests/test_document.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip("No sample PDF found at tests/sample.pdf")

    loader=PyPDFLoader(pdf_path)
    docs=loader.load()
    splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks=splitter.split_documents(docs)
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore=FAISS.from_documents(chunks,embeddings)
    llm=ChatOllama(model="mistral", temperature=0.0)
    return vectorstore, llm


def test_evaluate_without_responses():
    #CertainRAG in 2 signal mode - without semantic entropy
    vectorstore,llm=get_test_pipeline()

    query="What is the main topic of this document?"

    retrieved=vectorstore.similarity_search_with_score(query,k=4)
    context_chunks=[doc.page_content for doc,_ in retrieved]
    retrieval_scores=[float(score) for _, score in retrieved]
    retrieval_scores=normalize_l2(retrieval_scores)

    context_str="\n\n".join(context_chunks)
    answer=llm.invoke(f"Answer using only this context:\n{context_str}\n\nQuestion: {query}").content
    # Developer uses CertainRAG as pure evaluator
    rag=CertainRAG()
    result=rag.evaluate(
        answer=answer,
        context_chunks=context_chunks,
        retrieval_scores=retrieval_scores
    )
    assert result.uncertainty_level in {"LOW", "MEDIUM", "HIGH"}
    assert 0.0 <= result.uncertainty_score <= 1.0
    assert isinstance(result.should_abstain, bool)
    assert result.latency_ms > 0


def test_evaluate_with_responses():
    #CertainRAG with all three signals including semantic entropy
    vectorstore, llm=get_test_pipeline()
    query="What is the main topic of this document?"
    retrieved=vectorstore.similarity_search_with_score(query, k=4)
    context_chunks=[doc.page_content for doc,_ in retrieved]
    retrieval_scores=[float(score) for _, score in retrieved]
    retrieval_scores=normalize_l2(retrieval_scores)
    context_str="\n\n".join(context_chunks)
    prompt=f"Answer using only this context:\n{context_str}\n\nQuestion:{query}"

    # Developer generates answer and multiple responses
    answer=llm.invoke(prompt).content
    llm_with_temp=llm.__class__(model="mistral", temperature=0.7)
    responses=[llm_with_temp.invoke(prompt).content for _ in range(5)]

    rag=CertainRAG()
    result=rag.evaluate(
        answer=answer,
        context_chunks=context_chunks,
        retrieval_scores=retrieval_scores,
        responses=responses
    )
    assert result.uncertainty_level in {"LOW", "MEDIUM", "HIGH"}
    assert result.semantic_entropy != 0.5  # was actually computed
def test_fast_mode_uses_custom_embedding_model():
    rag = CertainRAG(fast_mode=True, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    assert rag._entropy.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert rag._entropy.fast_mode == True