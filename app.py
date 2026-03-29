import gradio as gr
from dotenv import load_dotenv
from app.loader import load_pdf
from app.chunking import recursive_chunking
from app.vectorstore import chunks_into_vector, load_vectorstore
from app.retriever import create_hybrid_retriever
from app.llm import create_llm, create_embeddings
from langchain_core.messages import HumanMessage, SystemMessage
import os

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """
You are a helpful AI assistant.
Answer ONLY from the given context.
If answer is not in context, say I don't know.
Context:
{context}
"""

# Global variables
retriever = None
llm       = None

def process_pdf(pdf_file):
    """Load and index uploaded PDF"""
    global retriever, llm

    text       = load_pdf(pdf_file.name)
    embeddings = create_embeddings()
    chunks     = recursive_chunking(text)
    db_path    = 'chroma_db_gradio'

    if os.path.exists(db_path):
        vectorstore = load_vectorstore(db_path, embeddings)
    else:
        vectorstore = chunks_into_vector(chunks, embeddings, db_path)

    retriever = create_hybrid_retriever(chunks, vectorstore)
    llm       = create_llm()

    return f"✅ PDF processed — {len(chunks)} chunks created. You can now ask questions."


def answer_question(question, history):
    """Answer question using RAG pipeline"""
    if retriever is None:
        return "Please upload a PDF first."

    docs          = retriever.invoke(question)
    context       = '\n\n'.join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response      = llm.invoke([
        SystemMessage(system_prompt),
        HumanMessage(question)
    ])
    return response.content


# Gradio Interface
with gr.Blocks(title="AI Research Assistant") as demo:

    gr.Markdown("# 📚 AI Research Assistant")
    gr.Markdown("Upload a PDF and ask questions about it using RAG pipeline with hybrid search.")

    with gr.Row():
        pdf_input    = gr.File(label="Upload PDF", file_types=[".pdf"])
        process_btn  = gr.Button("Process PDF", variant="primary")

    process_output = gr.Textbox(label="Status")
    process_btn.click(process_pdf, inputs=pdf_input, outputs=process_output)

    gr.Markdown("---")

    chatbot = gr.ChatInterface(
        fn=answer_question,
        title="Ask Questions About Your PDF"
    )

demo.launch(share=True)