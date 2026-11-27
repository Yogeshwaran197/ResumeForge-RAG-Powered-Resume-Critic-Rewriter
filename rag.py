from aiohttp import client_middleware_digest_auth
import pathlib  
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.retrievers import BM25Retriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("INVALID COHERE API KEY")

def  pdf_loader():

    path = input("Paste the path of your pdf : ").strip().strip('"')

    if not os.path.exists(path):
        raise ValueError("File path is invalid")

    loader = PyPDFLoader(path)

    try:
         doc = loader.load()
         resume_text = "\n".join(d.page_content for d in doc )
         print(f"Length of the document : {len(doc)}")

         splitter = RecursiveCharacterTextSplitter(
            chunk_size = 300,
            chunk_overlap = 100
         )
         chunks = splitter.split_documents(doc)
    except Exception as e:
        print(f"Error while loading resume: {e}")
        raise 

    return resume_text, chunks


def get_vectorstore(chunks):

    persist_memory = r"C:\Users\JANARTHAN\ResumeForge\ResumeForge-RAG-Powered-Resume-Critic-Rewriter"
    collection_name = "chroma_db"

    embedding = HuggingFaceEmbeddings(
        model_name ="BAAI/bge-m3"
    )
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            persist_directory=persist_memory,
            collection_name=collection_name,
            embedding= embedding
        )
    except Exception as error:
        print("Error while creating vectorstore {error}")
        raise

    return vectorstore


def hybrid_search():

    resume_text, chunks = pdf_loader()
    vectorstore = get_vectorstore(chunks)

    vector_search = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs = {
            "k" : 5,
            "fetch_k" : 10,
            "lambda_mult": 0.25
        }
    )

    bm25_search = BM25Retriever.from_documents(
        documents = chunks,
        k = 5
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_search, bm25_search],
        weights=[0.7,0.3]
    )

    cohere_rerank = CohereRerank(
        model = "rerank-v4.0-pro",
        top_n = 6,
        cohere_api_key = COHERE_API_KEY
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=cohere_rerank,
        base_retriever=ensemble_retriever
    )

    return retriever, resume_text





    



