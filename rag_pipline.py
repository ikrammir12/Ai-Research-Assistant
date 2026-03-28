from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage , SystemMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import time

#################
#1=load Dotrnv
#################
load_dotenv()

##################
#2=System_prompt
#################
SYSTEM_PROMPT_TEMPLATE = """
You are a helpful AI assistant.
Answer ONLY from the given context.
If answer is not in context, say "I don't know".

Context:
{context}
"""


#############
#3=load Pdf
#############

def load_pdf(pdf):
    text = PdfReader(pdf)
    full_text= ""
    for page in text.pages:
        full_text += '\n'+page.extract_text()

    return full_text


#######################################
#                                     #
# 4=Dividing the text into the chunks #
#     NOW WITH 3 STRATEGIES           #
#######################################

def fixed_chunking(text):
    """Strategy 1 - Cuts every N characters, no respect for meaning"""
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separator="\n"
    )
    return splitter.create_documents([text])


def recursive_chunking(text):
    """Strategy 2 - Tries paragraph → sentence → word → character"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.create_documents([text])


def semantic_chunking(text, embeddings):
    """Strategy 3 - Groups by meaning using embeddings"""
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70
    )
    return splitter.create_documents([text])


def compare_strategies(text, embeddings):
    """Run all 3 and print comparison table"""
    print("\n" + "="*50)
    print("CHUNKING STRATEGY COMPARISON")
    print("="*50)

    results = {}

    # Fixed
    start = time.time()
    fixed = fixed_chunking(text)
    results['Fixed Size'] = {
        'chunks': fixed,
        'count': len(fixed),
        'avg_size': sum(len(c.page_content) for c in fixed) // len(fixed),
        'time': round(time.time() - start, 3)
    }

    # Recursive
    start = time.time()
    recursive = recursive_chunking(text)
    results['Recursive'] = {
        'chunks': recursive,
        'count': len(recursive),
        'avg_size': sum(len(c.page_content) for c in recursive) // len(recursive),
        'time': round(time.time() - start, 3)
    }

    # Semantic
    start = time.time()
    semantic = semantic_chunking(text, embeddings)
    results['Semantic'] = {
        'chunks': semantic,
        'count': len(semantic),
        'avg_size': sum(len(c.page_content) for c in semantic) // len(semantic),
        'time': round(time.time() - start, 3)
    }

    # Print Table
    print(f"\n{'Strategy':<15} {'Chunks':<10} {'Avg Size':<12} {'Time'}")
    print("-" * 50)
    for name, data in results.items():
        print(f"{name:<15} {data['count']:<10} {data['avg_size']:<12} {data['time']}s")
    print("="*50)

    return results
    
#################
#               #
# 5=embeddding  #
#               #
#################
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

##################
#                #
# 6=vector store #
#                #
##################  
def chunks_into_vector(chunks,embeddings,db_path = 'chroma_db'):

    return Chroma.from_documents(embedding=embeddings,documents=chunks,
                                 persist_directory=db_path)


def load_vectorstore(db_path,embeddings):
    return Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

####################
#                  #
# Hybrid Retriever #
#                  #
####################

def create_hybrid_retriever(chunks, vectorstore):
    bm25_Retriever = BM25Retriever.from_documents(chunks)
    bm25_Retriever.k = 3

    vector_retriever = vectorstore.as_retriever(
        search_kwargs={'k':3}
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_Retriever,vector_retriever],
        weights=[0.5,0.5]
    )
    return hybrid_retriever

######################
#                    #
# compare retrievers #
#                    #
######################
def compare_retrievers(query, chunks,vectorstore):
    print('\n'+'='*60)
    print(f'QUERY :{query}')
    print('='*60)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k=3

    print('\n BM25 Results: ')
    for i, doc in enumerate(bm25.invoke(query)):
        print(f" {i+1}: {doc.page_content[:120]}")

    vector = vectorstore.as_retriever(search_kwargs={'k':3})
    print('Vector Results ') 
    for i , doc in enumerate(vector.invoke(query)):
        print(f" {i+1}: {doc.page_content[:120]}....")

    hybrid =EnsembleRetriever(
        retrievers=[bm25,vector],
        weights=[0.5,0.5]
    )
    print('\n hybrid results :') 
    for i , doc in enumerate(hybrid.invoke(query)):
        print(f" {i+1}: {doc.page_content[:120]}...")

    print('='*60)          
################
#              #
#  Create LLm  #
#              #
################

def create_llm():
    return ChatOpenAI(
        base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
        api_key=os.getenv('GEMINI_API_KEY'),
        model='gemini-2.5-flash',
        temperature=0.7
    )
##################################
#                                #
# 10 = Question Answer funcation #
#                                #
##################################
def question_answer(query,retriever,llm):
    retrival = retriever.invoke(query)
    context = '\n\n'.join(doc.page_content for doc in retrival)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke([SystemMessage(system_prompt),HumanMessage(query)]
                          )
    return response.content


##########################
#                        #
#  Main Pipline calling  #
#                        #
##########################

if __name__ == '__main__':

    pdf_path = r"D:\project\simple_rag_pipline\Simple_rag_pipeline\Modern Application Design and Development (CS-4103)_B.pdf"

    # Step 1: Extract text
    text = load_pdf(pdf_path)
    print(f"PDF loaded — Total characters: {len(text)}")

    # Step 2: Load embeddings first (needed for semantic chunking)
    embeddings = create_embeddings()

    # Step 3: Compare all 3 strategies
    results = compare_strategies(text, embeddings)

    # Step 4: Pick your strategy
    # Change "recursive" to "fixed" or "semantic" to test others
    STRATEGY = "recursive"

    if STRATEGY == "fixed":
        chunks = results['Fixed Size']['chunks']
    elif STRATEGY == "recursive":
        chunks = results['Recursive']['chunks']
    elif STRATEGY == "semantic":
        chunks = results['Semantic']['chunks']

    print(f"\n✅ Using strategy: {STRATEGY}")
    print(f"✅ Total chunks going into vector store: {len(chunks)}")

    # Step 5: Store and load
    db_path = f'chroma_db_{STRATEGY}'  # different db for each strategy
    if os.path.exists(db_path):
        vectorstore = load_vectorstore(db_path, embeddings)
        print('Loaded existing DB')
    else:
        vectorstore = chunks_into_vector(chunks, embeddings, db_path)
        print('Created new vector DB')

    # Step 6: Retriever
    #retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    retriever = create_hybrid_retriever(chunks,vectorstore)
    print('hybrid serach ready')
    # Step 7: LLM
    llm = create_llm()

    # Step 8: Ask questions
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        answer = question_answer(query, retriever, llm)
        print('\nAnswer:\n', answer)


