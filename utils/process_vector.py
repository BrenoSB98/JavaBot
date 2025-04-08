import os

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

persist_directory = 'db'

def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OllamaEmbeddings(),
        )
        return vector_store
    return None

def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(),
            persist_directory=persist_directory,
        )
    return vector_store