"""Build the tools + RouterQueryEngine"""

import sys
sys.path.append("./project")
sys.path.append("..")  # Add project root 

from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector

def router_query_engine(summary_index, vector_index):
    summary_query_engine = summary_index.as_query_engine()
    vector_query_engine = vector_index.as_query_engine()

    # Define tool selector/dispatch
    list_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
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

    return query_engine