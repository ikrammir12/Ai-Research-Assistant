from langchain_chroma import Chroma

def chunks_into_vector(chunks, embeddings, db_path='chroma_db'):
    """Store chunks in ChromaDB"""
    return Chroma.from_documents(
        embedding=embeddings,
        documents=chunks,
        persist_directory=db_path
    )

def load_vectorstore(db_path, embeddings):
    """Load existing ChromaDB"""
    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )