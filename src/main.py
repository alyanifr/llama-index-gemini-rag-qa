import os
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

# ==================== API KEY CALLS =====================

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# ========================================================

# Initializing reader and fetch data
reader = SimpleWebPageReader(html_to_text=True)
documents = reader.load_data(urls=["https://blog.google/innovation-and-ai/technology/research/technology-global-crisis-resilience/"])

# Initializing Gemini embeddings model
embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=GOOGLE_API_KEY,
    embedding_config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=200)
)

# Initialize Gemini
llm = GoogleGenAI(model="gemini-3.1-flash-lite", api_key=GOOGLE_API_KEY)

# Storing data using Chroma's vector
client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = client.get_or_create_collection("quickstart")

# Create vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Global settings
Settings.llm = llm
Settings.embed_model = embed_model

# Index the documents
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Creating retriever & generator 
# Load from disk
load_client = chromadb.PersistentClient(path="./chroma_db")

# Fetch the collection
collection = load_client.get_collection("quickstart")

# Fetch vector store
store = ChromaVectorStore(chroma_collection=collection)

# Get index from vector store
fetch_index = VectorStoreIndex.from_vector_store(vector_store=store)

# Initialized query engines
summary_query_engine = fetch_index.as_query_engine()
vector_query_engine = fetch_index.as_query_engine()

# Define tool selector/dispatcher
list_tool = QueryEngineTool.from_defaults(
    query_engine= summary_query_engine,
    description="Use this only when asked to summarize," \
    "describe, or give an overview of the article on the webpage." \
    "example; 'What is the article is about?' or 'Summarize the article/webpage'.",
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Use this for specific, detailed question about particular facts," \
    "figures, names, dates, or information mentioned in the article on the webpage." \
    "example; 'What did the article say about X?' or 'What was the date mentioned?'.",
)

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool
    ],
    verbose=True
)

response = query_engine.query("How did they accelerate disaster response?")
print(response)

