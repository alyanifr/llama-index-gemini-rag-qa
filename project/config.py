"""This file contains the API key calls, model settings and constants"""

import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

llm = GoogleGenAI(
    model="gemini-3.1-flash-lite",
    api_key=GOOGLE_API_KEY,
    temperature=0,
)

embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=GOOGLE_API_KEY,
    embedding_config=EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=200,
    ),
)

def configure_llama_index():
    """Call this once, before building any index — sets the global LLM
    and embedding model so as_query_engine() picks them up correctly."""
    Settings.llm = llm
    Settings.embed_model = embed_model

# Constants
CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "persist_collection"
