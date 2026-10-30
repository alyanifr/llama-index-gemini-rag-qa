# Evaluation summary

Source: `https://blog.google/innovation-and-ai/technology/research/technology-global-crisis-resilience/`  
Generated: 2026-10-30 16:12

Questions: 32. Scored under every variant: 32. Excluded because at least one variant errored: 0.

## Routing (paired questions only)

| variant | held-out (headline) | heldout_broad | heldout_specific | testset | dev_probe | all scored |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 8/16 | 0/8 | 8/8 | 10/10 | 0/6 | 18/32 |
| improved | 16/16 | 8/8 | 8/8 | 10/10 | 6/6 | 32/32 |

`held-out (headline)` combines the held-out broad and specific questions. `dev_probe` questions resemble the examples written into the improved tool descriptions, so they are shown for continuity but should not be quoted as the result.

### Where routing went

| variant | expected_tool | chosen_tool | count |
| --- | --- | --- | --- |
| baseline | summary_tool | vector_tool | 14 |
| baseline | vector_tool | vector_tool | 18 |
| improved | summary_tool | summary_tool | 14 |
| improved | vector_tool | vector_tool | 18 |

### Misrouted questions

| variant | label_source | question | chosen_tool |
| --- | --- | --- | --- |
| baseline | heldout_broad | Give me the gist of this page. | vector_tool |
| baseline | heldout_broad | In a few sentences, what should I take away from this? | vector_tool |
| baseline | heldout_broad | Walk me through the article from start to finish. | vector_tool |
| baseline | heldout_broad | What is the big picture here? | vector_tool |
| baseline | heldout_broad | Can you condense this into a short recap? | vector_tool |
| baseline | heldout_broad | What is the overall argument or message? | vector_tool |
| baseline | heldout_broad | TL;DR? | vector_tool |
| baseline | heldout_broad | Explain what this article covers as a whole. | vector_tool |
| baseline | dev_probe | Summarize this article. | vector_tool |
| baseline | dev_probe | What is this page about? | vector_tool |
| baseline | dev_probe | Give me an overview of the article. | vector_tool |
| baseline | dev_probe | What are the main points of this page? | vector_tool |
| baseline | dev_probe | Can you summarize the webpage in a few sentences? | vector_tool |
| baseline | dev_probe | What are the main themes of the article? | vector_tool |

## Answer quality (paired questions only)

| variant | similarity (matched) | similarity (control) | gap | n scored | context relevance mean | context relevance max | heuristic pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.974 | 0.868 | 0.107 | 10 | 0.812 | 0.820 | 32/32 |
| improved | 0.974 | 0.868 | 0.107 | 10 | 0.825 | 0.834 | 32/32 |

`similarity (control)` scores the same answers against a different question's reference. If it sits close to `similarity (matched)`, the gap is small and the metric is not separating right answers from unrelated ones.

## How to read this

- **Small sample.** Report counts (k/n), not just percentages, and treat a one-question difference as noise.
- **Expected routes are labels I assigned**, not ground truth: Ragas 'abstract' questions map to the summary tool, 'specific' ones to the vector tool, and the fixed question sets are hand-labelled. Check the borderline ones in `eval_results.csv`.
- **'baseline' is a reconstruction** of the original vague-description failure mode, not the exact historical text.
- **Both variants send the same test-set question to the same tool** unless routing differs, so answer-quality columns can only differ through routing. Identical numbers there are expected.
- **Context relevance** is embedding similarity between question and chunk, a proxy rather than an LLM judgement. It is only computed for vector-routed questions.
- **Semantic similarity** needs a reference answer, so the fixed question sets have none.
