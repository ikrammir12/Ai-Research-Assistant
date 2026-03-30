from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def create_hybrid_retriever(chunks, vectorstore):
    """Combine BM25 + Vector search"""
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    print("✅ Hybrid retriever ready")
    return hybrid_retriever


def compare_retrievers(query, chunks, vectorstore):
    """Show what each retriever finds"""
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 3

    print("\n📌 BM25 Results:")
    for i, doc in enumerate(bm25.invoke(query)):
        print(f"  {i+1}: {doc.page_content[:120]}")

    vector = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("\n🔍 Vector Results:")
    for i, doc in enumerate(vector.invoke(query)):
        print(f"  {i+1}: {doc.page_content[:120]}")

    hybrid = EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.5, 0.5]
    )
    print("\n⚡ Hybrid Results:")
    for i, doc in enumerate(hybrid.invoke(query)):
        print(f"  {i+1}: {doc.page_content[:120]}")
    print("="*60)