from typing import List, Tuple

def calculate_hallucination_index(classified_claims: List[Tuple[str, str]]) -> float:    
    if not classified_claims:
        return 0.0
    num_contradictions = sum(1 for _, label in classified_claims if label == "contradiction")
    total_claims = len(classified_claims)
    return num_contradictions / total_claims


def calculate_precision_recall(
    retrieved_docs: List[str], relevant_docs: List[str]
) -> Tuple[float, float]:
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    true_positives = len(retrieved_set & relevant_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    return precision, recall