from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import os

def create_llm():
    """Initialize Gemini LLM"""
    return ChatOpenAI(
        base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
        api_key=os.getenv('GEMINI_API_KEY'),
        model='gemini-2.5-flash',
        temperature=0.7
    )

def create_embeddings():
    """Initialize HuggingFace embeddings"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )