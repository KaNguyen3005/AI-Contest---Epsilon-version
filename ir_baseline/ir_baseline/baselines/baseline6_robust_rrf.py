"""
Baseline 6 - Robust RRF sparse ensemble.

This baseline is designed for the public-train/private-test setup:
it does not use answer labels inside retrieval, does not tune per-query
cutoffs from public answers, and avoids query-specific hand rules. It fuses
several sparse rankers with Reciprocal Rank Fusion (RRF), which is usually
more stable than picking the single best public-set model.
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from baselines.baseline2_bm25 import BM25Retriever
from baselines.baseline3_bm25_prf import PRFRetriever
from baselines.baseline4_bm25_ngram import EnhancedBM25Retriever
from baselines.baseline5_domain_expansion import DomainExpansionBM25Retriever


class RobustRRFEnsembleRetriever:
    def __init__(
        self,
        stem: bool = True,
        candidate_pool: int = 120,
        rrf_k: int = 60,
        bm25_weight: float = 1.00,
        prf_weight: float = 0.75,
        phrase_weight: float = 0.95,
        domain_weight: float = 0.85,
    ):
        self.stem = stem
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k
        self.rankers = [
            (
                "bm25",
                bm25_weight,
                BM25Retriever(k1=1.5, b=0.72, stem=stem),
            ),
            (
                "prf",
                prf_weight,
                PRFRetriever(
                    k1=1.5,
                    b=0.72,
                    prf_top_docs=3,
                    prf_top_terms=5,
                    alpha=1.0,
                    beta=0.45,
                    stem=stem,
                ),
            ),
            (
                "phrase",
                phrase_weight,
                EnhancedBM25Retriever(
                    k1=1.5,
                    b=0.72,
                    stem=stem,
                    max_ngram=3,
                    bigram_weight=0.10,
                    trigram_weight=0.14,
                    exact_phrase_boost=0.08,
                    candidate_pool=candidate_pool,
                ),
            ),
            (
                "domain",
                domain_weight,
                DomainExpansionBM25Retriever(
                    k1=1.5,
                    b=0.72,
                    stem=stem,
                    expansion_repeats=1,
                    rerank_pool=max(candidate_pool, 200),
                    low_speed_cylinder_boost=18.0,
                    toroidal_shell_boost=18.0,
                    stagnation_hypersonic_boost=10.0,
                ),
            ),
        ]

    def fit(self, corpus: Dict[int, str]):
        for name, _, ranker in self.rankers:
            print(f"[Robust RRF] Fitting {name} ...")
            ranker.fit(corpus)

    def _fused_ranking(self, text: str, pool_size: int) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = defaultdict(float)
        best_rank: Dict[int, int] = {}

        for _, weight, ranker in self.rankers:
            ranked = ranker.query(text, top_k=pool_size)
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                scores[doc_id] += weight / (self.rrf_k + rank)
                best_rank[doc_id] = min(rank, best_rank.get(doc_id, rank))

        fused = sorted(
            scores.items(),
            key=lambda item: (-item[1], best_rank[item[0]], item[0]),
        )
        return fused

    def query(self, text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        pool_size = max(self.candidate_pool, top_k * 20)
        return self._fused_ranking(text, pool_size=pool_size)[:top_k]

    def retrieve_all(
        self,
        queries: Dict[int, str],
        top_k: int = 5,
    ) -> Dict[int, List[int]]:
        results = {}
        for qid, qtext in queries.items():
            ranked = self.query(qtext, top_k=top_k)
            results[qid] = [doc_id for doc_id, _ in ranked]
        return results


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from utils.data_loader import load_answers, load_corpus, load_queries, save_submission
    from utils.evaluate import evaluate

    CORPUS_DIR = "data/Cranfield"
    QUERY_CSV = "data/public_test_queries.csv"
    ANSWER_CSV = "data/public_test_answers.csv"
    OUTPUT_CSV = "submissions/robust_rrf_submission.csv"

    corpus = load_corpus(CORPUS_DIR)
    queries = load_queries(QUERY_CSV)
    answers = load_answers(ANSWER_CSV)

    retriever = RobustRRFEnsembleRetriever()
    retriever.fit(corpus)
    results = retriever.retrieve_all(queries, top_k=5)

    print("\n=== Robust RRF Evaluation ===")
    evaluate(results, answers)

    os.makedirs("submissions", exist_ok=True)
    save_submission(results, OUTPUT_CSV)
