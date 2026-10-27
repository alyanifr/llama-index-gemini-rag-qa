"""Evaluate the router RAG pipeline.

For every question the pipeline records which tool the router picked, the
retrieved chunks and the answer, then scores:

* routing accuracy      - chosen tool vs expected tool (per description variant)
* semantic similarity   - cosine(answer, reference) on judge embeddings
* context relevance     - mean / max cosine(question, retrieved chunk)
                          (vector route only; the summary route reads every chunk)
* heuristic checks      - non-empty, minimum length, no hedge phrases

Typical usage (from the repo root, with GOOGLE_API_KEY set):

    python evaluation/eval_pipeline.py --url <page url>
    python evaluation/eval_pipeline.py --url <page url> --routing-only   # cheap
    python evaluation/eval_pipeline.py --dry-run                         # no API key

Outputs go to ./evaluation/data/: eval_results.csv and eval_summary.md
(prefixed with "dryrun_" for --dry-run, whose numbers are meaningless).
"""

import argparse
import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "project"))
sys.path.insert(0, str(ROOT / "evaluation"))

from llama_index.core import Settings  # noqa: E402
from llama_index.core.schema import QueryBundle  # noqa: E402

from project.indexing import build_indexes  # noqa: E402
from project.ingestion import chunk_document, load_document  # noqa: E402
from project.query_engine import (  # noqa: E402
    SUMMARY_TOOL_NAME,
    VARIANTS,
    VECTOR_TOOL_NAME,
    build_router_components,
)

DATA_DIR = ROOT / "evaluation" / "data"
DEFAULT_TESTSET = DATA_DIR / "generated_testset.csv"

MIN_ANSWER_CHARS = 40
HEDGE_PHRASES = (
    "i don't know", "i do not know", "cannot answer", "can't answer",
    "not mentioned", "no information", "does not provide", "not provided",
    "unable to", "context does not", "i'm sorry", "as an ai",
)

# Broad questions with a known expected route (summary tool).
BROAD_PROBES = (
    "Summarize this article.",
    "What is this page about?",
    "Give me an overview of the article.",
    "What are the main points of this page?",
    "Can you summarize the webpage in a few sentences?",
    "What are the main themes of the article?",
)


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------

def expected_tool_from_synthesizer(name: str) -> str:
    """Map a Ragas synthesizer to the tool that should answer it.

    This is a labelling assumption, not ground truth: abstract questions are
    treated as broad (summary tool); specific ones as detail lookups (vector
    tool). Add an `expected_tool` column to the CSV to override it.
    """
    return SUMMARY_TOOL_NAME if "abstract" in str(name).lower() else VECTOR_TOOL_NAME


def load_questions(testset: pd.DataFrame, include_probes: bool, limit: int | None) -> pd.DataFrame:
    df = testset.copy()
    if "expected_tool" not in df.columns:
        df["expected_tool"] = df["synthesizer_name"].map(expected_tool_from_synthesizer)
    df["label_source"] = "testset"
    df = df[["user_input", "reference", "expected_tool", "label_source"]]

    if include_probes:
        probes = pd.DataFrame(
            {
                "user_input": BROAD_PROBES,
                "reference": "",
                "expected_tool": SUMMARY_TOOL_NAME,
                "label_source": "broad_probe",
            }
        )
        df = pd.concat([df, probes], ignore_index=True)
    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


class Embedder:
    """Judge embeddings with a cache and light retry."""

    def __init__(self, model):
        self.model = model
        self.cache = {}

    def __call__(self, text: str):
        if text not in self.cache:
            self.cache[text] = with_retry(lambda: self.model.get_text_embedding(text))
        return self.cache[text]


def with_retry(fn, attempts: int = 3, base_delay: float = 2.0):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


def heuristic_checks(answer: str) -> dict:
    text = (answer or "").strip()
    lowered = text.lower()
    non_empty = bool(text)
    long_enough = len(text) >= MIN_ANSWER_CHARS
    hedged = any(p in lowered for p in HEDGE_PHRASES)
    return {
        "non_empty": non_empty,
        "long_enough": long_enough,
        "hedged": hedged,
        "heuristic_pass": non_empty and long_enough and not hedged,
    }


# --------------------------------------------------------------------------
# Running the router manually
# --------------------------------------------------------------------------

def run_question(selector, tools, question: str, embed: Embedder, reference: str, answer: bool) -> dict:
    """Route one question by calling the selector directly, then (optionally) answer it."""
    result = with_retry(
        lambda: selector.select([t.metadata for t in tools], QueryBundle(question))
    )
    selection = result.selections[0]
    tool = tools[selection.index]
    row = {
        "chosen_tool": tool.metadata.name,
        "route_reason": selection.reason,
    }
    if not answer:
        return row

    response = with_retry(lambda: tool.query_engine.query(question))
    answer_text = str(response)
    chunks = [n.node.get_content() for n in (response.source_nodes or [])]

    row["answer"] = answer_text
    row["n_chunks"] = len(chunks)
    row.update(heuristic_checks(answer_text))

    row["semantic_similarity"] = (
        cosine(embed(answer_text), embed(reference))
        if answer_text.strip() and str(reference).strip()
        else float("nan")
    )

    if tool.metadata.name == VECTOR_TOOL_NAME and chunks:
        q_vec = embed(question)
        sims = [cosine(q_vec, embed(c)) for c in chunks]
        row["context_relevance_mean"] = float(np.mean(sims))
        row["context_relevance_max"] = float(np.max(sims))
    else:
        row["context_relevance_mean"] = float("nan")
        row["context_relevance_max"] = float("nan")
    return row


def evaluate(summary_index, vector_index, questions: pd.DataFrame, embed: Embedder,
             variants, answer: bool) -> pd.DataFrame:
    rows = []
    for variant in variants:
        selector, tools = build_router_components(summary_index, vector_index, variant)
        for i, q in questions.iterrows():
            base = {"variant": variant, "question": q["user_input"],
                    "expected_tool": q["expected_tool"], "label_source": q["label_source"]}
            try:
                row = run_question(selector, tools, q["user_input"], embed, q["reference"], answer)
                row["error"] = ""
            except Exception as exc:  # keep going; record the failure
                row = {"error": f"{type(exc).__name__}: {exc}"}
            row = {**base, **row}
            row["route_correct"] = row.get("chosen_tool") == row["expected_tool"]
            rows.append(row)
            print(f"[{variant}] {i + 1}/{len(questions)} -> {row.get('chosen_tool', 'ERROR')}"
                  f" (expected {row['expected_tool']})", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def summarise(results: pd.DataFrame, answered: bool, dry_run: bool, url: str) -> str:
    ok = results[results["error"] == ""]
    out = ["# Evaluation summary", ""]
    if dry_run:
        out += ["> **DRY RUN: stub LLM and stub embeddings. These numbers only show that the "
                "pipeline runs; they say nothing about quality.**", ""]
    out += [f"Source: `{url}`  ", f"Generated: {time.strftime('%Y-%m-%d %H:%M')}", ""]

    # Routing
    rows = []
    for variant, g in results.groupby("variant", sort=False):
        valid = g[g["error"] == ""]
        row = {"variant": variant,
               "routing accuracy": f"{int(valid['route_correct'].sum())}/{len(valid)}"
                                   f" ({fmt(valid['route_correct'].mean(), 2)})",
               "errors": int((g["error"] != "").sum())}
        for src, sg in valid.groupby("label_source"):
            row[f"{src}"] = f"{int(sg['route_correct'].sum())}/{len(sg)}"
        rows.append(row)
    out += ["## Routing", "", md_table(pd.DataFrame(rows).fillna("-")), ""]

    # Confusion
    conf = (ok.groupby(["variant", "expected_tool", "chosen_tool"]).size()
              .reset_index(name="count"))
    out += ["### Where routing went", "", md_table(conf), ""]

    # Answer quality
    if answered and not ok.empty:
        rows = []
        for variant, g in ok.groupby("variant", sort=False):
            rows.append({
                "variant": variant,
                "semantic similarity (mean)": fmt(g["semantic_similarity"].mean()),
                "context relevance mean": fmt(g["context_relevance_mean"].mean()),
                "context relevance max": fmt(g["context_relevance_max"].mean()),
                "heuristic pass": f"{int(g['heuristic_pass'].sum())}/{len(g)}",
            })
        out += ["## Answer quality", "", md_table(pd.DataFrame(rows)), ""]

    out += [
        "## How to read this",
        "",
        "- **Small sample.** With a few dozen questions, a one- or two-question difference is noise. "
        "Report counts (k/n), not just percentages.",
        "- **Expected routes are labels I assigned**, not ground truth: Ragas 'abstract' questions "
        "map to the summary tool, 'specific' ones to the vector tool, and the broad probes are "
        "hand-written. Check the borderline ones in `eval_results.csv`.",
        "- **'baseline' is a reconstruction** of the original vague-description failure mode, not the "
        "exact historical text.",
        "- **Context relevance** is embedding similarity between question and chunk, a proxy rather "
        "than an LLM judgement. It is only computed for vector-routed questions.",
        "- **Semantic similarity** needs a reference answer, so broad probes have none.",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="Page to index (required unless --dry-run)")
    ap.add_argument("--testset", default=str(DEFAULT_TESTSET),
                    help="Ragas test set CSV (needs user_input, reference, synthesizer_name)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--routing-only", action="store_true",
                    help="Only record routing decisions (1 LLM call per question per variant)")
    ap.add_argument("--no-probes", action="store_true", help="Skip the built-in broad probes")
    ap.add_argument("--limit", type=int, help="Use only the first N questions")
    ap.add_argument("--dry-run", action="store_true", help="Offline stubs, no network or API key")
    args = ap.parse_args()

    if args.dry_run:
        from offline_stubs import SAMPLE_HTML, SAMPLE_TESTSET, HashEmbedding, KeywordRoutingLLM
        from llama_index.core import Document
        from project.ingestion import extract_paragraph_text

        Settings.llm = KeywordRoutingLLM()
        Settings.embed_model = HashEmbedding()
        judge = HashEmbedding()
        document = Document(text=extract_paragraph_text(SAMPLE_HTML))
        testset = pd.DataFrame(SAMPLE_TESTSET)
        url, prefix = "(built-in sample page)", "dryrun_"
    else:
        if not args.url:
            ap.error("--url is required unless --dry-run")
        from project.config import configure_llama_index, make_judge_embed_model

        configure_llama_index()
        judge = make_judge_embed_model()
        document = load_document(args.url)
        testset_path = Path(args.testset)
        if not testset_path.exists():
            ap.error(f"{testset_path} not found; generate it with evaluation/testset_generator.py")
        testset = pd.read_csv(testset_path)
        url, prefix = args.url, ""

    nodes = chunk_document(document)
    print(f"Indexed {len(nodes)} chunks.", flush=True)
    summary_index, vector_index = build_indexes(nodes)

    questions = load_questions(testset, include_probes=not args.no_probes, limit=args.limit)
    print(f"{len(questions)} questions x {len(args.variants)} variants.", flush=True)

    answered = not args.routing_only
    results = evaluate(summary_index, vector_index, questions, Embedder(judge),
                       args.variants, answer=answered)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"{prefix}eval_results.csv"
    md_path = DATA_DIR / f"{prefix}eval_summary.md"
    results.to_csv(csv_path, index=False)
    md_path.write_text(summarise(results, answered, args.dry_run, url), encoding="utf-8")
    print(f"Wrote {csv_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
