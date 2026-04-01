# 📚 AI Research Assistant

A production-grade RAG pipeline for querying PDF documents using 
hybrid search and evaluated with RAGAS metrics.

## 🏗️ Pipeline Architecture

PDF Upload → Chunking → Embeddings → Vector Store → Hybrid Search → LLM → Answer
                                                          ↑
                                                   BM25 + Vector

## ✨ Features

- 3 Chunking strategies (Fixed, Recursive, Semantic)
- Hybrid Search (BM25 + Vector with RRF)
- RAGAS Evaluation (Faithfulness, Relevancy, Context Recall)
- Gradio UI for easy interaction

## 📊 Evaluation Results

| Metric           | Score |
|------------------|-------|
| Faithfulness     | 1.000 |
| Answer Relevancy | 0.891 |
| Context Recall   | 1.000 |

## 🛠️ Tech Stack

- LangChain — pipeline framework
- ChromaDB — vector store
- HuggingFace — embeddings
- Gemini — LLM
- BM25 — keyword search
- RAGAS — evaluation
- Gradio — UI

## 🚀 Run Locally

pip install -r requirements.txt
python app.py