import sys
sys.path.insert(0, r'/Users/alyani/Documents/projects/GenAI Projects/llama-index-gemini-rag-qa')

import os
os.makedirs("./evaluation/data", exist_ok=True)

from ragas.testset import TestsetGenerator
from project.indexing import load_url
from project.config import llm, embed_model

website_url = "https://blog.google/innovation-and-ai/technology/research/technology-global-crisis-resilience/"

def testset_generator(website_url: str, llm, embed_model):

    # Initialize generator
    generator = TestsetGenerator.from_llama_index(
        llm=llm,
        embedding_model=embed_model,
    )

    # Generate testset
    print("Generating testset...")
    testset = generator.generate_with_llamaindex_docs(
        documents=load_url(website_url),
        testset_size=8,
    )

    # Save results to csv
    testset.to_pandas().to_csv("./evaluation/data/generated_testset.csv", index=False)
    print("Testset saved.")

    return testset

testset_generator(website_url=website_url, llm=llm, embed_model=embed_model)
print("Testset generated.")

