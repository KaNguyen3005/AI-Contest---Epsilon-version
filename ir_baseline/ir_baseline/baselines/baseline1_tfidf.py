"""
Baseline 1 — TF-IDF + Cosine Similarity
========================================
Cách hoạt động:
  1. Build inverted index từ corpus (TF-IDF weighting).
  2. Với mỗi query, tính TF-IDF vector của query.
  3. Tính cosine similarity giữa query vector và tất cả document vectors.
  4. Trả về top-K documents có similarity cao nhất.

Ưu điểm : Đơn giản, nhanh, hiệu quả tốt trên IR cổ điển.
Nhược điểm: Không xử lý được semantic similarity (từ đồng nghĩa).
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from utils.preprocessing import preprocess


class TFIDFRetriever:
    def __init__(self, stem: bool = True):
        self.stem = stem
        self.doc_ids: List[int] = []
        self.idf: Dict[str, float] = {}
        # doc_tfidf[doc_id][term] = tfidf weight
        self.doc_tfidf: Dict[int, Dict[str, float]] = {}
        # doc_norms[doc_id] = L2 norm of its TF-IDF vector
        self.doc_norms: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def fit(self, corpus: Dict[int, str]):
        """Build TF-IDF index from corpus."""
        print("[TF-IDF] Building index ...")
        N = len(corpus)
        self.doc_ids = sorted(corpus.keys())

        # Step 1: compute raw TF for each document
        raw_tf: Dict[int, Dict[str, int]] = {}
        df: Dict[str, int] = defaultdict(int)

        for doc_id, text in corpus.items():
            tokens = preprocess(text, stem=self.stem)
            counts: Dict[str, int] = defaultdict(int)
            for t in tokens:
                counts[t] += 1
            raw_tf[doc_id] = counts
            for term in counts:
                df[term] += 1

        # Step 2: compute IDF  = log((N+1) / (df+1)) + 1  (sklearn-style)
        self.idf = {
            term: math.log((N + 1) / (cnt + 1)) + 1
            for term, cnt in df.items()
        }

        # Step 3: compute normalised TF-IDF vectors
        for doc_id, counts in raw_tf.items():
            max_tf = max(counts.values()) if counts else 1
            tfidf: Dict[str, float] = {}
            for term, cnt in counts.items():
                tf = cnt / max_tf          # augmented TF (reduces bias toward long docs)
                tfidf[term] = tf * self.idf.get(term, 0.0)
            norm = math.sqrt(sum(v * v for v in tfidf.values())) or 1.0
            self.doc_tfidf[doc_id] = tfidf
            self.doc_norms[doc_id] = norm

        print(f"[TF-IDF] Indexed {N} documents, vocabulary size = {len(self.idf)}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def query(self, text: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Return top_k (doc_id, score) pairs sorted by cosine similarity."""
        tokens = preprocess(text, stem=self.stem)
        if not tokens:
            return []

        # Build query TF-IDF vector
        counts: Dict[str, int] = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        max_tf = max(counts.values())
        q_vec: Dict[str, float] = {
            term: (cnt / max_tf) * self.idf.get(term, 0.0)
            for term, cnt in counts.items()
        }
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        # Dot-product accumulation (only iterate non-zero query terms)
        scores: Dict[int, float] = defaultdict(float)
        for term, qw in q_vec.items():
            for doc_id in self.doc_ids:
                dw = self.doc_tfidf[doc_id].get(term, 0.0)
                if dw > 0:
                    scores[doc_id] += qw * dw

        # Normalise by L2 norms
        results = []
        for doc_id, dot in scores.items():
            cos_sim = dot / (q_norm * self.doc_norms[doc_id])
            results.append((doc_id, cos_sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def retrieve_all(
        self,
        queries: Dict[int, str],
        top_k: int = 20,
    ) -> Dict[int, List[int]]:
        """Run retrieval for all queries. Returns {query_id: [doc_id,...]}."""
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

    CORPUS_DIR  = "data/Cranfield"
    QUERY_CSV   = "data/public_test_queries.csv"
    ANSWER_CSV  = "data/public_test_answers.csv"
    OUTPUT_CSV  = "submissions/tfidf_submission.csv"

    corpus  = load_corpus(CORPUS_DIR)
    queries = load_queries(QUERY_CSV)
    answers = load_answers(ANSWER_CSV)

    retriever = TFIDFRetriever(stem=True)
    retriever.fit(corpus)

    results = retriever.retrieve_all(queries, top_k=20)

    print("\n=== TF-IDF Evaluation ===")
    evaluate(results, answers)

    os.makedirs("submissions", exist_ok=True)
    save_submission(results, OUTPUT_CSV)
