"""Streamlit UI only — thin, no business logic"""

import sys
sys.path.append("./project")
sys.path.append("..")  # Add project root 

import streamlit as st
from streamlit_chat import message
from config import configure_llama_index
from indexing import load_url, build_indexes
from query_engine import router_query_engine

# Streamlit page configuration & title
st.set_page_config(page_title="QA Bot", page_icon="📝", layout="centered")
st.title("🤖 WebPageReader: QnA Bot")

# Input option
website_url = st.text_input("Enter a URL", key="url")

# Only rebuild the pipeline when the URL actually changes
if website_url and st.session_state.get("processed_url") != website_url:
    with st.spinner("Reading & indexing the page..."):
        configure_llama_index()
        data = load_url(website_url)
        summary_index, vector_index = build_indexes(data)
        st.session_state.query_engine = router_query_engine(summary_index, vector_index)
        st.session_state.processed_url = website_url
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if website_url and (user_query := st.chat_input("Type your question here")):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    response = st.session_state.query_engine.query(user_query)
    st.session_state.messages.append({"role": "assistant", "content": str(response)})
    st.chat_message("assistant").write(str(response))


