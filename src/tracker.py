from typing import List, Tuple

from sentence_transformers import CrossEncoder


class HallucinationTracker:
    def __init__(
        self, nli_model_name: str = "cross-encoder/nli-deberta-v3-base"
    ):
        self.nli_model_name = nli_model_name
        self.nli_model = CrossEncoder(nli_model_name, device="cpu")

    def split_into_claims(self, response: str) -> List[str]:
        # Basic sentence splitting; can be improved with NLP libraries if needed
        claims = [s.strip() for s in response.split(".") if s.strip()]
        return claims

    def classify_claims(
        self, claims: List[str], context: str
    ) -> List[Tuple[str, str]]:
        
        label_map = {0: "contradiction", 1: "entailment", 2: "neutral"}
        pairs = [(context, claim) for claim in claims]
        predictions = self.nli_model.predict(pairs)
        labels = [label_map[pred.argmax()] for pred in predictions]
        return list(zip(claims, labels))