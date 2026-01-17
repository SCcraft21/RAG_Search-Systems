import os
from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.evaluator import calculate_hallucination_index, calculate_precision_recall
from src.retriever import Retriever
from src.tracker import HallucinationTracker

load_dotenv()

# Sample documents for demo (replace with your corpus)
SAMPLE_DOCUMENTS = [
    "The capital of France is Paris. It is known for the Eiffel Tower.",
    "Python is a popular programming language created by Guido van Rossum.",
    "Machine learning is a subset of artificial intelligence.",
    # Add more documents as needed
]

# Sample query for demo
SAMPLE_QUERY = "What is the capital of France?"

# Sample ground truth relevant docs for evaluation (replace as needed)
SAMPLE_RELEVANT_DOCS = ["The capital of France is Paris. It is known for the Eiffel Tower."]


def generate_response(llm, query: str, contexts: List[str]) -> str:
    context_str = "\n\n".join(contexts)
    prompt = PromptTemplate.from_template(
        "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"
    )
    chain = prompt | llm
    response = chain.invoke({"context": context_str, "question": query})
    return response.content


def main():
    # Initialize components (modular: swap model names as needed)
    retriever = Retriever()
    retriever.build_index(SAMPLE_DOCUMENTS)

    tracker = HallucinationTracker()

    # LLM setup (swap to Groq or HF as needed)
    # For Groq: from langchain_groq import ChatGroq; llm = ChatGroq(model="llama3-70b-8192")
    # For HF: from langchain_huggingface import HuggingFaceHub; llm = HuggingFaceHub(repo_id="meta-llama/Llama-3-8b", task="text-generation")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Run pipeline
    query = SAMPLE_QUERY
    retrieved_docs = retriever.retrieve(query)
    contexts = [doc.page_content for doc in retrieved_docs]
    context_str = " ".join(contexts)  # Concat for NLI

    response = generate_response(llm, query, contexts)

    claims = tracker.split_into_claims(response)
    classified_claims = tracker.classify_claims(claims, context_str)

    hi = calculate_hallucination_index(classified_claims)

    precision, recall = calculate_precision_recall(contexts, SAMPLE_RELEVANT_DOCS)

    # Print Factuality Report
    print("Factuality Report")
    print("=================")
    print(f"Query: {query}")
    print(f"Response: {response}")
    print("Retrieved Contexts:")
    for i, ctx in enumerate(contexts, 1):
        print(f"  {i}. {ctx}")
    print("Classified Claims:")
    for claim, label in classified_claims:
        print(f"  - {claim} ({label})")
    print(f"Hallucination Index (H_i): {hi:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")


if __name__ == "__main__":
    main()