"""This file contains webpage URL ingestion, build/load vector store and embediings indexing"""

import sys
sys.path.append("./project")
sys.path.append("..")  # Add project root 

import chromadb
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import SummaryIndex, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import CHROMA_PATH, CHROMA_COLLECTION_NAME

# Initializing reader 
reader = SimpleWebPageReader(html_to_text=True)

def load_url(url: str):
    """Fetch data ready to embed."""
    documents = reader.load_data(urls=[url])

    return documents

def build_indexes(documents):

    # Initializing chroma client and collection
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)

    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Indexing
    summary_index = SummaryIndex.from_documents(documents, storage_context=storage_context)
    vector_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    return summary_index, vector_index
