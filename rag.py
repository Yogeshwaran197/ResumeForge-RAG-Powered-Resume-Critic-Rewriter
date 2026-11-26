from torch import embedding
import pathlib  
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



def  pdf_loader():

    path = input("Paste the path of your pdf : ")

    if not os.exists(path):
        raise ValueError("File path is invalid")

    loader = PyPDFLoader()

    try:
         doc = loader.load(path)
         print("Length of the document")

         splitter = RecursiveCharacterTextSplitter(
            chunk_size = 300,
            chunk_overlap = 100
         )
         chunks = splitter.split_text(doc)
    except Exception as error:
        print("Error while loading loading Resume")

    return chunks



    



