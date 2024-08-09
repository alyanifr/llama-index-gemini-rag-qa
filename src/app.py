import os
from dotenv import load_dotenv
import urllib.request as url


import chromadb
import langchain
import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, SummaryIndex, download_loader
from llama_index.core import PromptTemplate
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configure gemini-pro api key
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-1.5-pro")

# Configure streamlit application
st.set_page_config(
    page_title="QA Bot",
    page_icon="📝",
    layout="centered"
)


# Streamlit page
st.title("🤖 WebPageReader: QnA Bot")

# Create a url input option
website_url = st.text_input("Enter a URL", key="url")

if website_url:
    try:
        # Initialize SimpleWebPageReader with the provided website url
        url_loader = SimpleWebPageReader()
        web_documents = url_loader.load_data(urls=[website_url])

        # Extract the content from the website
        html_content = web_documents[0].text

        # Parse the extracted content
        soup = BeautifulSoup(html_content, "html.parser")
        p_tags = soup.findAll("p")
        text_content = ""
        for each in p_tags:
            text_content += each.text + "\n"

        # Convert extracted text back to llama-index document format
        documents = [Document(text=text_content)]

        # Define sentence splitter and nodes
        splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=250)
        nodes = splitter.get_nodes_from_documents(documents)

        # Initialize gemini's embedding model
        gemini_embedding_model = GeminiEmbedding(model_name="models/embedding-001")

        # Initialize llm model
        llm = Gemini(model_name="models/gemini-1.5-flash-latest", temperature=0, streaming=True)

        # Create a chroma client and new collection
        client = chromadb.PersistentClient(path="./chromadb")
        chroma_collection = client.get_or_create_collection(name="persist_collection")

        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create a storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Set global settings
        Settings.llm = llm
        Settings.embed_model = gemini_embedding_model

        # Define summary index and vector index over the same data
        summary_index = SummaryIndex(nodes, storage_context=storage_context)
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

        # Create a retriever using chroma
        # Initialize the persistent client, load from disk
        load_client = chromadb.PersistentClient(path="./chromadb")

        # Fetch the collection
        chroma_collection = load_client.get_collection("persist_collection")

        # Fetch the vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Define query engines and set metadata
        summary_query_engines = summary_index.as_query_engine(
            response_mode="tree_summarize",
        )

        vector_query_engines = vector_index.as_query_engine()

        from llama_index.core.tools import QueryEngineTool

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engines,
            description=(
                f"Useful for summarizations of query related to context from {web_documents}"
            ),
        )

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engines,
            description=(
                f"Useful for retrieving specific answer for query from the context provided in {web_documents}"
            ),
        )

        # Define router query engines
        from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
        from llama_index.core.selectors import LLMSingleSelector

        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[summary_tool,vector_tool],
            verbose=True
        )

        # Configure streamlit chat_message
        message("Hello! How may I help?")

        if "message_history" not in st.session_state:
            st.session_state.message_history = []

        # Get user input for the query
        user_query = st.chat_input("Type your question here", key="input")

        if user_query:
            # Query the Langchain agent with user agent
            message(user_query, is_user=True)
            response = query_engine.query(user_query)

            # Display all previous messages
            for message_ in st.session_state.message_history:
                message(message_, is_user=True)

                # Placeholder for latest message
                placeholder = st.empty()
                input_ = st.text_input("You")
                st.session_state.message_history.append(input_)

                # Display latest message
                with placeholder.container():
                    message(st.session_state.message_history[-1], is_user=True)

            # Display the response
            st.text("Response:")
            message(str(response))

    except Exception as e:
        st.error(f"An error occurred: {e}")

