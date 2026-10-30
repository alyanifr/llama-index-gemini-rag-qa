"""This file contains webpage URL ingestion, build/load vector store and embediings indexing"""

import sys
sys.path.append("./project")
sys.path.append("..")  # Add project root 

import chromadb
# from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import SummaryIndex, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import CHROMA_COLLECTION_NAME

def build_indexes(nodes):
    """Return (summary_index, vector_index) for one page.

    EphemeralClient keeps everything in memory, so every call starts from an
    empty collection. A PersistentClient with a fixed collection name made
    chunks from earlier URLs leak into answers for the current one.
    """
    client = chromadb.EphemeralClient()
    # Ephemeral clients share one in-process backend, so make sure the
    # collection is fresh even if this function was called before.
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # The summary index keeps nodes in the docstore only; the vector index
    # is the one backed by Chroma.
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    
    return summary_index, vector_index

