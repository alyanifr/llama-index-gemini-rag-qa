"""Offline stand-ins used ONLY by `eval_pipeline.py --dry-run`.

They let the whole pipeline (scrape -> chunk -> index -> route -> answer ->
metrics -> report) run without network access or an API key, to check the
plumbing. Numbers produced with these stubs are meaningless as quality
measurements.
"""

import json
import re
import zlib
from typing import Any, List

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)

SAMPLE_HTML = """
<html><body>
<nav><p>Home</p><p>Products</p></nav>
<article>
<p>Researchers at the institute published a report describing a new flood forecasting system that covers more than eighty countries and gives warnings up to seven days ahead.</p>
<p>The system combines satellite imagery, river gauge readings and a machine learning model trained on decades of historical flood events across many river basins.</p>
<p>According to the report, the forecasts reached roughly four hundred million people in the first year, and local governments used them to plan evacuations earlier than before.</p>
<p>The team said the model is weaker in regions with few river gauges, and that it plans to add community reporting to improve coverage in those places during the next two years.</p>
<p>Funding for the project came from a mix of public grants and private donations, and the underlying data is available to researchers who apply for access through the institute.</p>
<p>The lead author explained that the main goal is to make flood warnings as routine as weather forecasts, so that communities can act before water arrives rather than after.</p>
</article>
<footer><p>Cookie settings</p></footer>
</body></html>
"""

SAMPLE_TESTSET = [
    {
        "user_input": "How many countries does the flood forecasting system cover?",
        "reference": "The system covers more than eighty countries.",
        "synthesizer_name": "single_hop_specific_query_synthesizer",
    },
    {
        "user_input": "How many people did the forecasts reach in the first year?",
        "reference": "Roughly four hundred million people.",
        "synthesizer_name": "single_hop_specific_query_synthesizer",
    },
    {
        "user_input": "Where is the model weaker and what will the team do about it?",
        "reference": "It is weaker where few river gauges exist; the team plans to add community reporting.",
        "synthesizer_name": "multi_hop_specific_query_synthesizer",
    },
    {
        "user_input": "What overall themes does the report develop about flood warnings?",
        "reference": "Making flood warnings routine and early so communities can act before water arrives.",
        "synthesizer_name": "multi_hop_abstract_query_synthesizer",
    },
]


class KeywordRoutingLLM(CustomLLM):
    """Answers selector prompts by keyword and everything else with a canned line."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=8192, num_output=256, model_name="offline-stub")

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        match = re.search(r"question: '(.*?)'\s*(?:\n|$)", prompt, flags=re.DOTALL)
        if match:  # selector prompt: choice 1 = summary tool, 2 = vector tool
            question = match.group(1).lower()
            broad = re.search(r"summar|overview|about overall|main points|main themes|overall theme|what is this", question)
            choice = 1 if broad else 2
            return CompletionResponse(
                text=json.dumps([{"choice": choice, "reason": "offline keyword stub"}])
            )
        return CompletionResponse(
            text="Offline stub answer built from the supplied context passages of the article."
        )

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        yield self.complete(prompt, formatted=formatted, **kwargs)


class HashEmbedding(BaseEmbedding):
    """Deterministic bag-of-words hashing embedding (64 dims)."""

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(64)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vec[zlib.crc32(token.encode()) % 64] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)
