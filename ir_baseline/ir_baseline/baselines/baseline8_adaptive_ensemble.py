"""
Baseline 8 - Adaptive ProfileKNN + BM25 ensemble.

The public set rewards profile matching because many queries are paraphrases
with shared answer sets. Synthetic tests show that pure profile matching can
fall behind BM25 when queries are less clustered. This ensemble uses the
nearest-profile similarity as a confidence signal: high confidence leans
toward ProfileKNN; low confidence falls back toward BM25.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from baselines.baseline7_profile_knn import ProfileKNNRetriever


class AdaptiveProfileBM25Ensemble:
    def __init__(
        self,
        stem: bool = True,
        neighbors: int = 8,
        min_similarity: float = 0.08,
        bm25_pool: int = 80,
        low_confidence: float = 0.30,
        high_confidence: float = 0.55,
        min_profile_weight: float = 0.65,
        max_profile_weight: float = 0.98,
        disagreement_profile_weight: float = 0.20,
    ):
        self.profile = ProfileKNNRetriever(
            stem=stem,
            neighbors=neighbors,
            min_similarity=min_similarity,
            profile_weight=1.0,
            lexical_weight=0.0,
            bm25_pool=bm25_pool,
        )
        self.bm25_pool = bm25_pool
        self.low_confidence = low_confidence
        self.high_confidence = high_confidence
        self.min_profile_weight = min_profile_weight
        self.max_profile_weight = max_profile_weight
        self.disagreement_profile_weight = disagreement_profile_weight
        self.train_answers: Dict[int, List[int]] = {}

    def fit(
        self,
        corpus: Dict[int, str],
        train_queries: Optional[Dict[int, str]] = None,
        train_answers: Optional[Dict[int, List[int]]] = None,
    ):
        self.train_answers = train_answers or {}
        self.profile.fit(corpus, train_queries=train_queries, train_answers=train_answers)

    def _adaptive_profile_weight(self, confidence: float) -> float:
        if confidence <= self.low_confidence:
            return self.min_profile_weight
        if confidence >= self.high_confidence:
            return self.max_profile_weight
        span = self.high_confidence - self.low_confidence
        ratio = (confidence - self.low_confidence) / span if span > 0 else 1.0
        return self.min_profile_weight + ratio * (self.max_profile_weight - self.min_profile_weight)

    @staticmethod
    def _normalise(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        max_score = max(scores.values()) or 1.0
        return {doc_id: score / max_score for doc_id, score in scores.items()}

    def _profile_scores(
        self,
        text: str,
        exclude_qids: Optional[Set[int]] = None,
    ) -> Tuple[Dict[int, float], float]:
        scores: Dict[int, float] = defaultdict(float)
        neighbours = self.profile._nearest_profiles(text, exclude_qids=exclude_qids)
        confidence = neighbours[0][1] if neighbours else 0.0

        for neighbour_rank, (train_qid, sim) in enumerate(neighbours, start=1):
            neighbour_decay = 1.0 / (neighbour_rank ** 0.5)
            for answer_rank, doc_id in enumerate(self.train_answers.get(train_qid, []), start=1):
                answer_decay = 1.0 / (answer_rank ** 0.5)
                scores[doc_id] += sim * neighbour_decay * answer_decay

        return self._normalise(scores), confidence

    def _bm25_scores(self, text: str, top_k: int) -> Dict[int, float]:
        ranked = self.profile.bm25.query(text, top_k=max(self.bm25_pool, top_k))
        scores: Dict[int, float] = {}
        max_bm25 = ranked[0][1] if ranked else 1.0
        max_bm25 = max_bm25 or 1.0
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            rank_decay = 1.0 / (rank ** 0.5)
            scores[doc_id] = (score / max_bm25) * rank_decay
        return self._normalise(scores)

    def query(
        self,
        text: str,
        top_k: int = 5,
        exclude_qids: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        profile_scores, confidence = self._profile_scores(text, exclude_qids=exclude_qids)
        bm25_scores = self._bm25_scores(text, top_k=top_k)

        profile_top = {
            doc_id
            for doc_id, _ in sorted(profile_scores.items(), key=lambda item: -item[1])[:top_k]
        }
        bm25_top = {
            doc_id
            for doc_id, _ in sorted(bm25_scores.items(), key=lambda item: -item[1])[:top_k]
        }
        agreement = len(profile_top & bm25_top) / top_k if top_k > 0 else 0.0

        if agreement == 0.0:
            profile_weight = self.disagreement_profile_weight
        elif agreement < 0.2 and confidence < self.high_confidence:
            profile_weight = min(self.min_profile_weight, 0.35)
        else:
            profile_weight = self._adaptive_profile_weight(confidence)
        bm25_weight = 1.0 - profile_weight

        doc_ids = set(profile_scores) | set(bm25_scores)
        fused = [
            (
                doc_id,
                profile_weight * profile_scores.get(doc_id, 0.0)
                + bm25_weight * bm25_scores.get(doc_id, 0.0),
            )
            for doc_id in doc_ids
        ]
        fused.sort(key=lambda item: (-item[1], item[0]))
        return fused[:top_k]

    def retrieve_all(
        self,
        queries: Dict[int, str],
        top_k: int = 5,
        leave_one_out: bool = True,
    ) -> Dict[int, List[int]]:
        results = {}
        for qid, qtext in queries.items():
            exclude = {qid} if leave_one_out and qid in self.train_answers else set()
            ranked = self.query(qtext, top_k=top_k, exclude_qids=exclude)
            results[qid] = [doc_id for doc_id, _ in ranked]
        return results
