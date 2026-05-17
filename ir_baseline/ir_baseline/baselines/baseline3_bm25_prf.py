"""
Baseline 3 — BM25 + Pseudo-Relevance Feedback (PRF / Rocchio)
===============================================================
Cách hoạt động:
  Pseudo-Relevance Feedback (PRF) là kỹ thuật query expansion không cần nhãn:
    1. Chạy BM25 lần 1 → lấy top-R documents làm "pseudo-relevant".
    2. Trích xuất top-M terms quan trọng nhất từ pseudo-relevant docs
       (dùng relevance model / TF-IDF weighting).
    3. Mở rộng query gốc với các terms mới (Rocchio algorithm).
    4. Chạy BM25 lần 2 với expanded query → kết quả cuối cùng.

  Rocchio expansion:
    q_expanded = α * q_original + β * (mean TF-IDF of top-R docs)

  Hyperparameters:
    prf_top_docs  = 5   (số pseudo-relevant docs)
    prf_top_terms = 10  (số expansion terms thêm vào)
    alpha = 1.0         (trọng số query gốc)
    beta  = 0.8         (trọng số feedback terms)

Ưu điểm : Thường cải thiện recall đáng kể so với BM25 thuần.
Nhược điểm: Nếu top docs ban đầu sai (query drift), có thể giảm precision.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from utils.preprocessing import preprocess
from baselines.baseline2_bm25 import BM25Retriever


class PRFRetriever:
    def __init__(
        self,
        k1: float = 1.5,
        b:  float = 0.75,
        prf_top_docs:  int   = 5,
        prf_top_terms: int   = 10,
        alpha:         float = 1.0,
        beta:          float = 0.8,
        stem:          bool  = True,
    ):
        self.bm25 = BM25Retriever(k1=k1, b=b, stem=stem)
        self.stem = stem
        self.prf_top_docs  = prf_top_docs
        self.prf_top_terms = prf_top_terms
        self.alpha         = alpha
        self.beta          = beta

        # Will be filled after fit()
        self.idf:      Dict[str, float] = {}
        self.tf:       Dict[int, Dict[str, int]] = {}
        self.doc_lens: Dict[int, int] = {}

    def fit(self, corpus: Dict[int, str]):
        self.bm25.fit(corpus)
        # Share the index
        self.idf      = self.bm25.idf
        self.tf       = self.bm25.tf
        self.doc_lens = self.bm25.doc_lens

    # ------------------------------------------------------------------
    # Rocchio expansion
    # ------------------------------------------------------------------
    def _expand_query(
        self,
        query_tokens: List[str],
        pseudo_doc_ids: List[int],
    ) -> Dict[str, float]:
        """
        Returns an expanded query as a weighted term dict.
        q_exp[term] = alpha * q_weight + beta * feedback_weight
        """
        # --- Original query vector (count-based) ---
        q_counts: Dict[str, int] = defaultdict(int)
        for t in query_tokens:
            q_counts[t] += 1
        max_qtf = max(q_counts.values()) if q_counts else 1

        q_vec: Dict[str, float] = {
            t: self.alpha * (cnt / max_qtf) * self.idf.get(t, 0.0)
            for t, cnt in q_counts.items()
        }

        # --- Feedback vector: mean TF-IDF over pseudo-relevant docs ---
        fb_vec: Dict[str, float] = defaultdict(float)
        R = len(pseudo_doc_ids)
        if R > 0:
            for doc_id in pseudo_doc_ids:
                dl = self.doc_lens[doc_id]
                tf_d = self.tf[doc_id]
                dl_safe = dl if dl > 0 else 1
                for term, cnt in tf_d.items():
                    tfidf = (cnt / dl_safe) * self.idf.get(term, 0.0)
                    fb_vec[term] += tfidf / R

        # Keep only top-M feedback terms (by weight), exclude query terms
        existing = set(q_vec.keys())
        new_terms = sorted(
            [(t, w) for t, w in fb_vec.items() if t not in existing],
            key=lambda x: x[1], reverse=True
        )[:self.prf_top_terms]

        # Merge
        expanded: Dict[str, float] = dict(q_vec)
        for term, weight in new_terms:
            expanded[term] = self.beta * weight

        return expanded

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def query(self, text: str, top_k: int = 20) -> List[Tuple[int, float]]:
        tokens = preprocess(text, stem=self.stem)
        if not tokens:
            return []

        # Pass 1: BM25 with original query
        pass1 = self.bm25.query(text, top_k=self.prf_top_docs)
        pseudo_ids = [doc_id for doc_id, _ in pass1]

        # Expand query
        expanded = self._expand_query(tokens, pseudo_ids)

        # Pass 2: score ALL docs with expanded query
        # We treat expanded as a weighted query and use a generalised BM25 score
        k1, b, avgdl = self.bm25.k1, self.bm25.b, self.bm25.avgdl

        candidate_ids: set = set()
        for term in expanded:
            for doc_id, tf_d in self.tf.items():
                if tf_d.get(term, 0) > 0:
                    candidate_ids.add(doc_id)

        if not candidate_ids:
            return pass1[:top_k]

        results = []
        for doc_id in candidate_ids:
            dl   = self.doc_lens[doc_id]
            tf_d = self.tf[doc_id]
            score = 0.0
            for term, qw in expanded.items():
                if term not in self.idf:
                    continue
                f = tf_d.get(term, 0)
                if f == 0:
                    continue
                num = f * (k1 + 1)
                den = f + k1 * (1 - b + b * dl / avgdl)
                score += qw * (num / den)
            results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def retrieve_all(
        self,
        queries: Dict[int, str],
        top_k: int = 20,
    ) -> Dict[int, List[int]]:
        results = {}
        for qid, qtext in queries.items():
            ranked = self.query(qtext, top_k=top_k)
            results[qid] = [doc_id for doc_id, _ in ranked]
        return results


# ------------------------------------------------------------------
# Quick standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from utils.data_loader import load_corpus, load_queries, load_answers, save_submission
    from utils.evaluate import evaluate

    CORPUS_DIR = "data/Cranfield"
    QUERY_CSV  = "data/public_test_queries.csv"
    ANSWER_CSV = "data/public_test_answers.csv"
    OUTPUT_CSV = "submissions/prf_submission.csv"

    corpus  = load_corpus(CORPUS_DIR)
    queries = load_queries(QUERY_CSV)
    answers = load_answers(ANSWER_CSV)

    retriever = PRFRetriever(
        prf_top_docs=5, prf_top_terms=10,
        alpha=1.0, beta=0.8, stem=True
    )
    retriever.fit(corpus)

    results = retriever.retrieve_all(queries, top_k=20)

    print("\n=== BM25 + PRF Evaluation ===")
    evaluate(results, answers)

    os.makedirs("submissions", exist_ok=True)
    save_submission(results, OUTPUT_CSV)
