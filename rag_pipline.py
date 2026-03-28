from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage , SystemMessage
from langchain_core.utils.utils import convert_to_secret_str

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
#                                     #
#######################################
def text_into_chunks(text):
    spliter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    
    return spliter.create_documents([text])
    
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

if __name__=='__main__':
    
    pdf_path = r"D:\project\simple_rag_pipline\Simple_rag_pipeline\Modern Application Design and Development (CS-4103)_B.pdf"

    #step 1: Extract text 
    text = load_pdf(pdf_path)
    
    #step 2: Creating chunks
    chunks = text_into_chunks(text)

    #step 3 : Embeddings
    embeddings = create_embeddings()

    #step 4:Store and load
    db_path = 'chroma_db'
    if os.path.exists(db_path):
        vectorstore = load_vectorstore(db_path,embeddings)
        print('Loaded Exiting DB')
    else:
        vectorstore = chunks_into_vector(chunks,embeddings,db_path)
        print('created the new vector db')

    #step 5: Retriever
    retriever = vectorstore.as_retriever(search_kwargs={'k':3})

    #step 6: LLM
    llm = create_llm()

    #step 7 : As question

    while True:
        query = input("\nAsk a question (or type 'exit'):") 

        if query.lower() =='exit':
            break
        answer = question_answer(query,retriever,llm) 
        print('\n Answer \n',answer)



