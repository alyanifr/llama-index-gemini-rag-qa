import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                    # makes `project` importable
sys.path.insert(0, str(ROOT / "evaluation")) 

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ragas_compat  # noqa: F401  (must come before ragas)

from ragas.testset import TestsetGenerator
from project.ingestion import load_document
from project.config import llm, embed_model

DEFAULT_URL = "https://blog.google/innovation-and-ai/technology/research/technology-global-crisis-resilience/"
OUTPUT = ROOT / "evaluation" / "data" / "generated_testset.csv"

def testset_generator(website_url: str, testset_size: int = 10):

    # Initialize generator
    generator = TestsetGenerator.from_llama_index(
        llm=llm,
        embedding_model=embed_model,
    )

    # Generate testset
    print("Generating testset...")
    testset = generator.generate_with_llamaindex_docs(
        documents=[load_document(website_url)],
        testset_size=testset_size,
    )

    # Save results to csv
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    testset.to_pandas().to_csv(OUTPUT, index=False)
    print(f"Testset saved to {OUTPUT}")

    return testset

if __name__ == "__main__":
    testset_generator(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL)
