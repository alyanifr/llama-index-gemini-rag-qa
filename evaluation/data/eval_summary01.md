# Evaluation summary

Source: `https://blog.google/innovation-and-ai/technology/research/technology-global-crisis-resilience/`  
Generated: 2026-10-30 15:38

## Routing

| variant | routing accuracy | errors | broad_probe | testset |
| --- | --- | --- | --- | --- |
| baseline | 10/14 (0.71) | 2 | 0/4 | 10/10 |
| improved | 14/14 (1.00) | 2 | 6/6 | 8/8 |

### Where routing went

| variant | expected_tool | chosen_tool | count |
| --- | --- | --- | --- |
| baseline | summary_tool | vector_tool | 4 |
| baseline | vector_tool | vector_tool | 10 |
| improved | summary_tool | summary_tool | 6 |
| improved | vector_tool | vector_tool | 8 |

## Answer quality

| variant | semantic similarity (mean) | context relevance mean | context relevance max | heuristic pass |
| --- | --- | --- | --- | --- |
| baseline | 0.974 | 0.837 | 0.844 | 14/14 |
| improved | 0.974 | 0.851 | 0.857 | 14/14 |

## How to read this

- **Small sample.** With a few dozen questions, a one- or two-question difference is noise. Report counts (k/n), not just percentages.
- **Expected routes are labels I assigned**, not ground truth: Ragas 'abstract' questions map to the summary tool, 'specific' ones to the vector tool, and the broad probes are hand-written. Check the borderline ones in `eval_results.csv`.
- **'baseline' is a reconstruction** of the original vague-description failure mode, not the exact historical text.
- **Context relevance** is embedding similarity between question and chunk, a proxy rather than an LLM judgement. It is only computed for vector-routed questions.
- **Semantic similarity** needs a reference answer, so broad probes have none.
