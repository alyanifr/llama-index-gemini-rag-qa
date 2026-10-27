"""Build the tools and the RouterQueryEngine.

Two sets of tool descriptions live here on purpose:

* "baseline"  - a reconstruction of the original failure mode: two vague,
                near-identical descriptions, so the selector LLM has little to
                separate them.
* "improved"  - explicit "ONLY for broad..." / "for specific..." framing with
                example questions.

The evaluation pipeline runs both through the same question set so the effect
of the description rewrite can be measured instead of asserted.
"""

import sys
sys.path.append("./project")
sys.path.append("..")  # Add project root 

from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.selectors import LLMSingleSelector

SUMMARY_TOOL_NAME = "summary_tool"
VECTOR_TOOL_NAME = "vector_tool"

DESCRIPTIONS = {
    "baseline": {
        SUMMARY_TOOL_NAME: "Useful for answering questions about the web page.",
        VECTOR_TOOL_NAME: "Useful for answering questions about the content of the web page.",
    },
    "improved": {
        SUMMARY_TOOL_NAME: (
            "Use this ONLY for broad, high-level questions that need the whole page: "
            "summaries, overviews, main themes, or what the page is about overall. "
            "Examples: 'Summarize this article.', 'What is this page about?', "
            "'What are the main points?'. Do NOT use it for questions about a "
            "specific fact, name, number or date."
        ),
        VECTOR_TOOL_NAME: (
            "Use this for specific, detailed questions whose answer sits in one or "
            "two passages: particular facts, figures, names, dates, definitions, or "
            "what the page says about one narrow topic. Examples: 'What date was "
            "mentioned for the launch?', 'What did the article say about flood "
            "forecasting?', 'Who is quoted?'."
        ),
    },
}

VARIANTS = tuple(DESCRIPTIONS)


def build_router_components(summary_index, vector_index, variant: str = "improved"):
    """Return (selector, tools) so callers can inspect routing decisions.

    `tools` is ordered [summary_tool, vector_tool]; the selector's returned
    index refers to that order.
    """
    if variant not in DESCRIPTIONS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
    descriptions = DESCRIPTIONS[variant]

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(response_mode="tree_summarize"),
        name=SUMMARY_TOOL_NAME,
        description=descriptions[SUMMARY_TOOL_NAME],
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(),
        name=VECTOR_TOOL_NAME,
        description=descriptions[VECTOR_TOOL_NAME],
    )
    
    selector = LLMSingleSelector.from_defaults()
    return selector, [summary_tool, vector_tool]


def router_query_engine(summary_index, vector_index, variant: str = "improved"):
    """The router used by the Streamlit app."""
    selector, tools = build_router_components(summary_index, vector_index, variant)
    return RouterQueryEngine(selector=selector, query_engine_tools=tools, verbose=True)