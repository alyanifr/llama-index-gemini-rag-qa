"""Evaluate the router RAG pipeline.

For every question the pipeline records which tool the router picked, the
retrieved chunks and the answer, then scores:

* routing accuracy      - chosen tool vs expected tool (per description variant)
* semantic similarity   - cosine(answer, reference) on judge embeddings, next to
                          a control: the same answers scored against a *different*
                          question's reference. A small gap means the metric
                          cannot tell right answers from wrong ones.
* context relevance     - mean / max cosine(question, retrieved chunk)
                          (vector route only; the summary route reads every chunk)
* heuristic checks      - non-empty, minimum length, no hedge phrases

Fair comparison rules
* Paired scoring: a question counts only if it succeeded under EVERY variant, so
  an API hiccup in one run cannot change which questions each variant is judged on.
  Failed rows are retried with backoff, then re-run once after a cooldown.
* Held-out questions: the headline routing number uses broad and specific
  questions worded differently from the examples written into the improved tool
  descriptions. The original dev probes are reported separately for continuity.

Typical usage (from the repo root, with GOOGLE_API_KEY set):

    python evaluation/eval_pipeline.py --url <page url>
    python evaluation/eval_pipeline.py --url <page url> --routing-only   # cheap
    python evaluation/eval_pipeline.py --url <page url> --pause 2        # gentler on rate limits
    python evaluation/eval_pipeline.py --dry-run                         # no API key

Outputs go to ./evaluation/data/: eval_results.csv and eval_summary.md
(prefixed with "dryrun_" for --dry-run, whose numbers are meaningless).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
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

# Retry behaviour; overridden from the command line in main().
RETRIES = 5          # attempts per call
BASE_DELAY = 4.0     # seconds; doubles each attempt: 4, 8, 16, 32
PAUSE = 0.0          # seconds to wait after every question
RERUN_PASSES = 1     # extra passes over rows that still failed
COOLDOWN = 30.0      # seconds to wait before a re-run pass

# --------------------------------------------------------------------------
# Question sets (page-agnostic, so they work for any URL)
# --------------------------------------------------------------------------

# The original probes. Several resemble example questions written inside the
# improved tool descriptions, so they are NOT a fair test of that variant.
DEV_PROBES = (
    "Summarize this article.",
    "What is this page about?",
    "Give me an overview of the article.",
    "What are the main points of this page?",
    "Can you summarize the webpage in a few sentences?",
    "What are the main themes of the article?",
)

# Broad questions worded to avoid the description examples and keywords
# ("summarize", "about", "main points", "overview", "main themes").
HELDOUT_BROAD = (
    "Give me the gist of this page.",
    "In a few sentences, what should I take away from this?",
    "Walk me through the article from start to finish.",
    "What is the big picture here?",
    "Can you condense this into a short recap?",
    "What is the overall argument or message?",
    "TL;DR?",
    "Explain what this article covers as a whole.",
)

# Specific lookups, including two phrased like broad questions. Labels are my
# judgement: check the borderline ones in eval_results.csv.
HELDOUT_SPECIFIC = (
    "Which organizations are named in the article?",
    "What numbers or statistics does the article cite?",
    "Who is quoted, and what exactly do they say?",
    "Does the article mention any dates or years? Which ones?",
    "What does the article say about funding or cost?",
    "Which locations or countries are mentioned?",
    "What is the single most important statistic in the article?",
    "What does the article say about the tools or products it mentions?",
)

SOURCE_ORDER = ("heldout_broad", "heldout_specific", "testset", "dev_probe")


def expected_tool_from_synthesizer(name: str) -> str:
    """Map a Ragas synthesizer to the tool that should answer it.

    This is a labelling assumption, not ground truth: abstract questions are
    treated as broad (summary tool); specific ones as detail lookups (vector
    tool). Add an `expected_tool` column to the CSV to override it.
    """
    return SUMMARY_TOOL_NAME if "abstract" in str(name).lower() else VECTOR_TOOL_NAME


def _fixed(questions, expected_tool, label_source) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_input": list(questions),
            "reference": "",
            "expected_tool": expected_tool,
            "label_source": label_source,
        }
    )


def load_questions(testset: pd.DataFrame, dev_probes: bool, heldout: bool,
                   limit: int | None) -> pd.DataFrame:
    df = testset.copy()
    if "expected_tool" not in df.columns:
        df["expected_tool"] = df["synthesizer_name"].map(expected_tool_from_synthesizer)
    df["label_source"] = "testset"
    df = df[["user_input", "reference", "expected_tool", "label_source"]]

    parts = [df]
    if heldout:
        parts.append(_fixed(HELDOUT_BROAD, SUMMARY_TOOL_NAME, "heldout_broad"))
        parts.append(_fixed(HELDOUT_SPECIFIC, VECTOR_TOOL_NAME, "heldout_specific"))
    if dev_probes:
        parts.append(_fixed(DEV_PROBES, SUMMARY_TOOL_NAME, "dev_probe"))
    out = pd.concat(parts, ignore_index=True)
    if limit:
        out = out.head(limit)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


def with_retry(fn, attempts: int | None = None, base_delay: float | None = None):
    attempts = attempts or RETRIES
    base_delay = BASE_DELAY if base_delay is None else base_delay
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


class Embedder:
    """Judge embeddings with a cache and retry."""

    def __init__(self, model):
        self.model = model
        self.cache = {}

    def __call__(self, text: str):
        if text not in self.cache:
            self.cache[text] = with_retry(lambda: self.model.get_text_embedding(text))
        return self.cache[text]


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


def add_similarity_control(results: pd.DataFrame, embed: Embedder) -> pd.DataFrame:
    """Score each answer against the NEXT question's reference.

    If this control lands close to the real similarity, the metric is not
    separating correct answers from unrelated ones.
    """
    results["similarity_control"] = np.nan
    for _, g in results.groupby("variant", sort=False):
        ok = g[
            g["error"].fillna("").eq("")
            & g["reference"].fillna("").astype(str).str.strip().ne("")
            & g["answer"].fillna("").astype(str).str.strip().ne("")
        ]
        if len(ok) < 2:
            continue
        idx = list(ok.index)
        refs = [str(r) for r in ok["reference"]]
        for k, i in enumerate(idx):
            other = refs[(k + 1) % len(idx)]
            results.at[i, "similarity_control"] = cosine(embed(str(ok.at[i, "answer"])), embed(other))
    return results


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

    row["reference"] = reference
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
    components = {v: build_router_components(summary_index, vector_index, v) for v in variants}

    def attempt(variant: str, i: int) -> dict:
        q = questions.loc[i]
        selector, tools = components[variant]
        base = {"variant": variant, "qid": i, "question": q["user_input"],
                "expected_tool": q["expected_tool"], "label_source": q["label_source"]}
        try:
            row = run_question(selector, tools, q["user_input"], embed, q["reference"], answer)
            row["error"] = ""
        except Exception as exc:  # keep going; record the failure
            row = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        row = {**base, **row}
        row["route_correct"] = row.get("chosen_tool") == row["expected_tool"]
        if PAUSE:
            time.sleep(PAUSE)
        return row

    rows = {}
    for variant in variants:
        for i in questions.index:
            rows[(variant, i)] = attempt(variant, i)
            r = rows[(variant, i)]
            print(f"[{variant}] {i + 1}/{len(questions)} -> {r.get('chosen_tool', 'ERROR')}"
                  f" (expected {r['expected_tool']})", flush=True)

    for pass_no in range(1, RERUN_PASSES + 1):
        failed = [k for k, r in rows.items() if r["error"]]
        if not failed:
            break
        print(f"Re-running {len(failed)} failed row(s) after a {COOLDOWN:.0f}s cooldown "
              f"(pass {pass_no}).", flush=True)
        time.sleep(COOLDOWN)
        for variant, i in failed:
            rows[(variant, i)] = attempt(variant, i)
            r = rows[(variant, i)]
            print(f"[{variant}] retry {i + 1} -> {r.get('chosen_tool', 'ERROR')}", flush=True)

    return pd.DataFrame(list(rows.values()))


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


def split_paired(results: pd.DataFrame):
    """Return (paired rows, excluded rows): paired = questions with no error in any variant."""
    results = results.copy()
    results["error"] = results["error"].fillna("")
    clean = results.groupby("qid")["error"].apply(lambda s: bool((s == "").all()))
    keep = clean[clean].index
    return results[results["qid"].isin(keep)], results[~results["qid"].isin(keep)]


def kn(g: pd.DataFrame) -> str:
    return f"{int(g['route_correct'].sum())}/{len(g)}" if len(g) else "-"


def summarise(results: pd.DataFrame, answered: bool, dry_run: bool, url: str) -> str:
    paired, excluded = split_paired(results)
    out = ["# Evaluation summary", ""]
    if dry_run:
        out += ["> **DRY RUN: stub LLM and stub embeddings. These numbers only show that the "
                "pipeline runs; they say nothing about quality.**", ""]
    out += [f"Source: `{url}`  ", f"Generated: {time.strftime('%Y-%m-%d %H:%M')}", ""]

    n_total = results["qid"].nunique()
    n_paired = paired["qid"].nunique()
    out += [f"Questions: {n_total}. Scored under every variant: {n_paired}. "
            f"Excluded because at least one variant errored: {n_total - n_paired}.", ""]

    # Routing (paired rows only)
    rows = []
    for variant, g in paired.groupby("variant", sort=False):
        held = g[g["label_source"].str.startswith("heldout")]
        row = {"variant": variant, "held-out (headline)": kn(held)}
        for src in SOURCE_ORDER:
            sg = g[g["label_source"] == src]
            if len(sg):
                row[src] = kn(sg)
        row["all scored"] = kn(g)
        rows.append(row)
    out += ["## Routing (paired questions only)", "", md_table(pd.DataFrame(rows).fillna("-")), ""]
    out += ["`held-out (headline)` combines the held-out broad and specific questions. "
            "`dev_probe` questions resemble the examples written into the improved tool "
            "descriptions, so they are shown for continuity but should not be quoted as the result.", ""]

    conf = (paired.groupby(["variant", "expected_tool", "chosen_tool"]).size()
                  .reset_index(name="count"))
    out += ["### Where routing went", "", md_table(conf), ""]

    # Misroutes, for eyeballing labels
    wrong = paired[~paired["route_correct"]][["variant", "label_source", "question", "chosen_tool"]]
    if len(wrong):
        out += ["### Misrouted questions", "", md_table(wrong.head(30)), ""]

    # Excluded
    if len(excluded):
        ex = (excluded[excluded["error"] != ""]
              .groupby(["qid", "question"])
              .apply(lambda g: "; ".join(f"{v}: {e[:60]}" for v, e in zip(g["variant"], g["error"])),
                     include_groups=False)
              .reset_index(name="errors"))
        out += ["### Excluded questions", "", md_table(ex.drop(columns="qid").head(20)), ""]

    # Answer quality (paired rows only)
    if answered and not paired.empty:
        rows = []
        for variant, g in paired.groupby("variant", sort=False):
            sim = g["semantic_similarity"]
            ctl = g["similarity_control"] if "similarity_control" in g else pd.Series(dtype=float)
            both = g[sim.notna() & ctl.notna()]
            rows.append({
                "variant": variant,
                "similarity (matched)": fmt(both["semantic_similarity"].mean()) if len(both) else "n/a",
                "similarity (control)": fmt(both["similarity_control"].mean()) if len(both) else "n/a",
                "gap": fmt((both["semantic_similarity"] - both["similarity_control"]).mean()) if len(both) else "n/a",
                "n scored": len(both),
                "context relevance mean": fmt(g["context_relevance_mean"].mean()),
                "context relevance max": fmt(g["context_relevance_max"].mean()),
                "heuristic pass": f"{int(g['heuristic_pass'].sum())}/{len(g)}",
            })
        out += ["## Answer quality (paired questions only)", "", md_table(pd.DataFrame(rows)), ""]
        out += ["`similarity (control)` scores the same answers against a different question's "
                "reference. If it sits close to `similarity (matched)`, the gap is small and the "
                "metric is not separating right answers from unrelated ones.", ""]

    out += [
        "## How to read this",
        "",
        "- **Small sample.** Report counts (k/n), not just percentages, and treat a one-question "
        "difference as noise.",
        "- **Expected routes are labels I assigned**, not ground truth: Ragas 'abstract' questions "
        "map to the summary tool, 'specific' ones to the vector tool, and the fixed question sets "
        "are hand-labelled. Check the borderline ones in `eval_results.csv`.",
        "- **'baseline' is a reconstruction** of the original vague-description failure mode, not the "
        "exact historical text.",
        "- **Both variants send the same test-set question to the same tool** unless routing differs, "
        "so answer-quality columns can only differ through routing. Identical numbers there are expected.",
        "- **Context relevance** is embedding similarity between question and chunk, a proxy rather "
        "than an LLM judgement. It is only computed for vector-routed questions.",
        "- **Semantic similarity** needs a reference answer, so the fixed question sets have none.",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    global RETRIES, BASE_DELAY, PAUSE, RERUN_PASSES, COOLDOWN

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="Page to index (required unless --dry-run)")
    ap.add_argument("--testset", default=str(DEFAULT_TESTSET),
                    help="Ragas test set CSV (needs user_input, reference, synthesizer_name)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--routing-only", action="store_true",
                    help="Only record routing decisions (1 LLM call per question per variant)")
    ap.add_argument("--no-heldout", action="store_true", help="Skip the held-out question sets")
    ap.add_argument("--no-dev-probes", action="store_true", help="Skip the original dev probes")
    ap.add_argument("--limit", type=int, help="Use only the first N questions")
    ap.add_argument("--retries", type=int, default=RETRIES, help="Attempts per API call")
    ap.add_argument("--pause", type=float, default=PAUSE, help="Seconds to wait after each question")
    ap.add_argument("--cooldown", type=float, default=COOLDOWN, help="Seconds before re-running failed rows")
    ap.add_argument("--rerun-passes", type=int, default=RERUN_PASSES, help="Extra passes over failed rows")
    ap.add_argument("--dry-run", action="store_true", help="Offline stubs, no network or API key")
    args = ap.parse_args()

    RETRIES, PAUSE = args.retries, args.pause
    COOLDOWN, RERUN_PASSES = args.cooldown, args.rerun_passes

    if args.dry_run:
        from offline_stubs import SAMPLE_HTML, SAMPLE_TESTSET, HashEmbedding, KeywordRoutingLLM
        from llama_index.core import Document
        from ingestion import extract_paragraph_text

        BASE_DELAY, COOLDOWN = 0.0, 0.0
        Settings.llm = KeywordRoutingLLM()
        Settings.embed_model = HashEmbedding()
        judge = HashEmbedding()
        document = Document(text=extract_paragraph_text(SAMPLE_HTML))
        testset = pd.DataFrame(SAMPLE_TESTSET)
        url, prefix = "(built-in sample page)", "dryrun_"
    else:
        if not args.url:
            ap.error("--url is required unless --dry-run")
        from config import configure_llama_index, make_judge_embed_model

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

    questions = load_questions(testset, dev_probes=not args.no_dev_probes,
                               heldout=not args.no_heldout, limit=args.limit)
    print(f"{len(questions)} questions x {len(args.variants)} variants.", flush=True)

    answered = not args.routing_only
    embed = Embedder(judge)
    results = evaluate(summary_index, vector_index, questions, embed, args.variants, answer=answered)
    if answered:
        results = add_similarity_control(results, embed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / f"{prefix}eval_results.csv"
    md_path = DATA_DIR / f"{prefix}eval_summary.md"
    results.to_csv(csv_path, index=False)
    md_path.write_text(summarise(results, answered, args.dry_run, url), encoding="utf-8")
    print(f"Wrote {csv_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()