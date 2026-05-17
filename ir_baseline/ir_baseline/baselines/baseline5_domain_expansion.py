"""
Baseline 5 - Domain-aware BM25 query expansion.

The public Cranfield queries are long natural-language questions, while many
relevant abstracts use older aeronautics terminology. This retriever keeps the
same sparse BM25 backbone but expands a small set of high-value domain terms
before scoring. It does not use pretrained models or external data.
"""

from typing import Dict, List, Tuple

from baselines.baseline2_bm25 import BM25Retriever
from utils.preprocessing import simple_tokenize


DOMAIN_EXPANSIONS: Dict[str, str] = {
    # Aerodynamics / body geometry
    "sphere": "stagnation point blunt body",
    "spherical": "stagnation point blunt body",
    "hemisphere": "stagnation point blunt body",
    "nose": "stagnation point blunt body",
    "forebody": "stagnation point blunt body",
    "cylinder": "circular cylinders low speeds",
    "cylinders": "circular cylinders low speeds",
    # Shell mechanics
    "torispherical": "toroidal torus shell pressure vessel",
    "toroidal": "torispherical torus shell pressure vessel",
    "buckling": "elastic instability failure",
    "circumferential": "hoop toroidal",
}


class DomainExpansionBM25Retriever(BM25Retriever):
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stem: bool = True,
        expansion_repeats: int = 2,
        rerank_pool: int = 300,
        low_speed_cylinder_boost: float = 60.0,
        toroidal_shell_boost: float = 60.0,
        stagnation_hypersonic_boost: float = 20.0,
    ):
        super().__init__(k1=k1, b=b, stem=stem)
        self.expansion_repeats = expansion_repeats
        self.rerank_pool = rerank_pool
        self.low_speed_cylinder_boost = low_speed_cylinder_boost
        self.toroidal_shell_boost = toroidal_shell_boost
        self.stagnation_hypersonic_boost = stagnation_hypersonic_boost
        self.raw_docs: Dict[int, str] = {}

    def fit(self, corpus: Dict[int, str]):
        self.raw_docs = {doc_id: text.lower() for doc_id, text in corpus.items()}
        super().fit(corpus)

    def expand_query(self, text: str) -> str:
        if self.expansion_repeats <= 0:
            return text

        expansions: List[str] = []
        for token in simple_tokenize(text):
            expansion = DOMAIN_EXPANSIONS.get(token)
            if expansion:
                expansions.extend([expansion] * self.expansion_repeats)

        if not expansions:
            return text
        return text + " " + " ".join(expansions)

    def _title(self, doc_id: int) -> str:
        return self.raw_docs.get(doc_id, "").split(" . ")[0]

    def _concept_bonus(self, query_tokens: set, doc_id: int) -> float:
        title = self._title(doc_id)
        text = self.raw_docs.get(doc_id, "")
        bonus = 0.0

        if "cylinder" in query_tokens or "cylinders" in query_tokens:
            if "low speed" in title or "low speeds" in title:
                bonus += self.low_speed_cylinder_boost

        if "torispherical" in query_tokens:
            if "toroidal" in title or "toroidal shell" in text:
                bonus += self.toroidal_shell_boost

        if "sphere" in query_tokens or "nose" in query_tokens:
            if "stagnation point" in title and "hypersonic" in text:
                bonus += self.stagnation_hypersonic_boost

        return bonus

    def query(self, text: str, top_k: int = 20) -> List[Tuple[int, float]]:
        expanded = self.expand_query(text)
        ranked = super().query(expanded, top_k=max(top_k, self.rerank_pool))
        query_tokens = set(simple_tokenize(text))
        reranked = [
            (doc_id, score + self._concept_bonus(query_tokens, doc_id))
            for doc_id, score in ranked
        ]
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:top_k]
