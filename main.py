from dotenv import load_dotenv
from app.loader import load_pdf
from app.chunking import compare_strategies
from app.vectorstore import chunks_into_vector, load_vectorstore
from app.retriever import create_hybrid_retriever
from app.llm import create_llm, create_embeddings
from app.evaluation import (
    create_test_dataset,
    run_rag_on_testset,
    run_ragas_evaluation,
    print_evaluation_results
)
from langchain_core.messages import HumanMessage, SystemMessage
import os
import gradio as gr
import warnings
import os
import logging

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("chromadb").setLevel(logging.ERROR)

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful AI assistant.
Answer ONLY from the given context.
If answer is not in context, say I don't know.
Context:
{context}
"""

def question_answer(query, retriever, llm):
    docs          = retriever.invoke(query)
    context       = '\n\n'.join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response      = llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(query)
    ])
    return response.content


if __name__ == '__main__':

    pdf_path = r"D:\project\Ai research\Ai-Research-Assistant\Modern Application Design and Development (CS-4103)_B.pdf"

    # Step 1 — Load
    text = load_pdf(pdf_path)

    # Step 2 — Embeddings
    embeddings = create_embeddings()

    # Step 3 — Chunking
    results_chunking = compare_strategies(text, embeddings)
    chunks = results_chunking['Recursive']['chunks']

    # Step 4 — Vector store
    db_path = 'chroma_db_recursive'
    if os.path.exists(db_path):
        vectorstore = load_vectorstore(db_path, embeddings)
        print("✅ Loaded existing DB")
    else:
        vectorstore = chunks_into_vector(chunks, embeddings, db_path)
        print("✅ Created new vector DB")

    # Step 5 — Hybrid retriever
    retriever = create_hybrid_retriever(chunks, vectorstore)

    # Step 6 — LLM
    llm = create_llm()
    print("✅ LLM ready")

    # Step 7 — RAGAS evaluation
    # Comment out after first run
    test_data   = create_test_dataset()
    rag_results = run_rag_on_testset(test_data, retriever, llm)
    scores      = run_ragas_evaluation(rag_results, llm, embeddings)
    df          = print_evaluation_results(scores)

    # Step 8 — QA loop
def answer(question, history):
    return question_answer(question, retriever, llm)

gr.ChatInterface(
    fn=answer,
    title="📚 AI Research Assistant",
    description="Upload context is pre-loaded. Ask questions about the document."
).launch(share=True)

