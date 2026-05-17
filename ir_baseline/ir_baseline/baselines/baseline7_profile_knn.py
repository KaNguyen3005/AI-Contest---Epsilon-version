"""
Baseline 7 - Supervised query-profile kNN.

Use this when public queries with public answers are allowed as training data.
For validation on the same public split, retrieve_all() excludes the current
query id from its own neighbour set, so the score is a leave-one-out estimate
instead of a memorised public-answer score.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from baselines.baseline2_bm25 import BM25Retriever
from utils.preprocessing import preprocess


class ProfileKNNRetriever:
    def __init__(
        self,
        stem: bool = True,
        neighbors: int = 8,
        min_similarity: float = 0.08,
        profile_weight: float = 0.80,
        lexical_weight: float = 0.20,
        bm25_pool: int = 80,
    ):
        self.stem = stem
        self.neighbors = neighbors
        self.min_similarity = min_similarity
        self.profile_weight = profile_weight
        self.lexical_weight = lexical_weight
        self.bm25_pool = bm25_pool
        self.bm25 = BM25Retriever(k1=1.5, b=0.72, stem=stem)
        self.train_answers: Dict[int, List[int]] = {}
        self.query_vectors: Dict[int, Dict[str, float]] = {}
        self.query_norms: Dict[int, float] = {}

    def fit(
        self,
        corpus: Dict[int, str],
        train_queries: Optional[Dict[int, str]] = None,
        train_answers: Optional[Dict[int, List[int]]] = None,
    ):
        self.bm25.fit(corpus)
        self.train_answers = train_answers or {}
        self.query_vectors = {}
        self.query_norms = {}

        if not train_queries or not train_answers:
            return

        for qid, text in train_queries.items():
            if qid not in train_answers:
                continue
            vec = self._query_vector(text)
            self.query_vectors[qid] = vec
            self.query_norms[qid] = self._norm(vec)

    def _query_vector(self, text: str) -> Dict[str, float]:
        counts: Dict[str, int] = defaultdict(int)
        for token in preprocess(text, stem=self.stem):
            counts[token] += 1
        return {
            token: (1.0 + math.log(count)) * self.bm25.idf.get(token, 0.0)
            for token, count in counts.items()
        }

    @staticmethod
    def _norm(vec: Dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vec.values())) or 1.0

    @staticmethod
    def _cosine(
        left: Dict[str, float],
        left_norm: float,
        right: Dict[str, float],
        right_norm: float,
    ) -> float:
        if len(left) > len(right):
            left, right = right, left
            left_norm, right_norm = right_norm, left_norm
        dot = sum(weight * right.get(term, 0.0) for term, weight in left.items())
        return dot / (left_norm * right_norm) if dot > 0 else 0.0

    def _nearest_profiles(
        self,
        text: str,
        exclude_qids: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        if not self.query_vectors:
            return []
        exclude_qids = exclude_qids or set()
        query_vec = self._query_vector(text)
        query_norm = self._norm(query_vec)

        neighbours = []
        for train_qid, train_vec in self.query_vectors.items():
            if train_qid in exclude_qids:
                continue
            sim = self._cosine(query_vec, query_norm, train_vec, self.query_norms[train_qid])
            if sim >= self.min_similarity:
                neighbours.append((train_qid, sim))

        neighbours.sort(key=lambda item: item[1], reverse=True)
        return neighbours[:self.neighbors]

    def query(
        self,
        text: str,
        top_k: int = 5,
        exclude_qids: Optional[Set[int]] = None,
    ) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = defaultdict(float)

        neighbours = self._nearest_profiles(text, exclude_qids=exclude_qids)
        for neighbour_rank, (train_qid, sim) in enumerate(neighbours, start=1):
            neighbour_decay = 1.0 / math.sqrt(neighbour_rank)
            for answer_rank, doc_id in enumerate(self.train_answers.get(train_qid, []), start=1):
                answer_decay = 1.0 / math.sqrt(answer_rank)
                scores[doc_id] += self.profile_weight * sim * neighbour_decay * answer_decay

        bm25_ranked = self.bm25.query(text, top_k=max(self.bm25_pool, top_k))
        max_bm25 = bm25_ranked[0][1] if bm25_ranked else 1.0
        max_bm25 = max_bm25 or 1.0
        for rank, (doc_id, score) in enumerate(bm25_ranked, start=1):
            rank_decay = 1.0 / math.sqrt(rank)
            scores[doc_id] += self.lexical_weight * (score / max_bm25) * rank_decay

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

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
