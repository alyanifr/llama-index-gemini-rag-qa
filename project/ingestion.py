"""Fetch a web page, extract its article text, and split it into chunks (nodes)."""

import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250
MIN_PARAGRAPH_CHARS = 40
REQUEST_TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (compatible; rag-qa-bot/1.0)"

def extract_paragraph_text(html: str) -> str:
    """Return the text of the page's <p> tags, one paragraph per line."""
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = (p.get_text(" ", strip=True) for p in soup.find_all("p"))
    return "\n\n".join(p for p in paragraphs if len(p) >= MIN_PARAGRAPH_CHARS)


def load_document(url: str) -> Document:
    """Scrape `url` (<p> tags only) into a single LlamaIndex Document.

    Limitation: content that lives outside <p> tags (tables, lists, headings,
    figure captions) is not indexed.
    """
    response = requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()

    text = extract_paragraph_text(response.text)
    if not text.strip():
        raise ValueError(f"No <p> text found at {url}")

    return Document(text=text, metadata={"source_url": url})


def chunk_document(document: Document):
    """Split a Document into nodes (size 1000, overlap 250)."""
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.get_nodes_from_documents([document])


def load_and_chunk_url(url: str):
    """Scrape `url` and return SentenceSplitter nodes ready for indexing."""
    return chunk_document(load_document(url))