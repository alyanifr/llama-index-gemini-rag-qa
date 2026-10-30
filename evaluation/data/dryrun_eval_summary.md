# Evaluation summary

> **DRY RUN: stub LLM and stub embeddings. These numbers only show that the pipeline runs; they say nothing about quality.**

Source: `(built-in sample page)`  
Generated: 2026-10-30 16:00

Questions: 26. Scored under every variant: 26. Excluded because at least one variant errored: 0.

## Routing (paired questions only)

| variant | held-out (headline) | heldout_broad | heldout_specific | testset | dev_probe | all scored |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 16/16 | 8/8 | 8/8 | 4/4 | 6/6 | 26/26 |
| improved | 16/16 | 8/8 | 8/8 | 4/4 | 6/6 | 26/26 |

`held-out (headline)` combines the held-out broad and specific questions. `dev_probe` questions resemble the examples written into the improved tool descriptions, so they are shown for continuity but should not be quoted as the result.

### Where routing went

| variant | expected_tool | chosen_tool | count |
| --- | --- | --- | --- |
| baseline | summary_tool | summary_tool | 15 |
| baseline | vector_tool | vector_tool | 11 |
| improved | summary_tool | summary_tool | 15 |
| improved | vector_tool | vector_tool | 11 |

## Answer quality (paired questions only)

| variant | similarity (matched) | similarity (control) | gap | n scored | context relevance mean | context relevance max | heuristic pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.328 | 0.328 | 0.000 | 4 | 0.462 | 0.462 | 26/26 |
| improved | 0.328 | 0.328 | 0.000 | 4 | 0.462 | 0.462 | 26/26 |

`similarity (control)` scores the same answers against a different question's reference. If it sits close to `similarity (matched)`, the gap is small and the metric is not separating right answers from unrelated ones.

## How to read this

- **Small sample.** Report counts (k/n), not just percentages, and treat a one-question difference as noise.
- **Expected routes are labels I assigned**, not ground truth: Ragas 'abstract' questions map to the summary tool, 'specific' ones to the vector tool, and the fixed question sets are hand-labelled. Check the borderline ones in `eval_results.csv`.
- **'baseline' is a reconstruction** of the original vague-description failure mode, not the exact historical text.
- **Both variants send the same test-set question to the same tool** unless routing differs, so answer-quality columns can only differ through routing. Identical numbers there are expected.
- **Context relevance** is embedding similarity between question and chunk, a proxy rather than an LLM judgement. It is only computed for vector-routed questions.
- **Semantic similarity** needs a reference answer, so the fixed question sets have none.
