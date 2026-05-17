"""
Baseline 2 — BM25 (Okapi BM25)
================================
Cách hoạt động:
  BM25 là thuật toán ranking IR cổ điển, cải tiến TF-IDF với:
    - TF saturation: tránh bias khi từ xuất hiện quá nhiều
    - Document length normalisation: cân bằng giữa doc dài/ngắn

  Score(q, d) = Σ IDF(t) * [tf(t,d) * (k1+1)] / [tf(t,d) + k1*(1 - b + b*|d|/avgdl)]

  Hyperparameters:
    k1 = 1.5  (TF saturation — higher = slower saturation)
    b  = 0.75 (length normalisation — 0: no norm, 1: full norm)

Ưu điểm : Thường outperform TF-IDF, rất ổn định trên các tập IR.
Nhược điểm: Vẫn là bag-of-words, không nắm được ngữ nghĩa.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from utils.preprocessing import preprocess


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75, stem: bool = True):
        self.k1 = k1
        self.b  = b
        self.stem = stem

        self.doc_ids:  List[int] = []
        self.idf:      Dict[str, float] = {}
        self.tf:       Dict[int, Dict[str, int]] = {}   # raw counts
        self.doc_lens: Dict[int, int] = {}
        self.avgdl:    float = 0.0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def fit(self, corpus: Dict[int, str]):
        print("[BM25] Building index ...")
        N = len(corpus)
        self.doc_ids = sorted(corpus.keys())

        df:  Dict[str, int] = defaultdict(int)
        total_len = 0

        for doc_id, text in corpus.items():
            tokens = preprocess(text, stem=self.stem)
            counts: Dict[str, int] = defaultdict(int)
            for t in tokens:
                counts[t] += 1
            self.tf[doc_id] = counts
            self.doc_lens[doc_id] = len(tokens)
            total_len += len(tokens)
            for term in counts:
                df[term] += 1

        self.avgdl = total_len / N if N else 1.0

        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)  — Robertson IDF
        self.idf = {
            term: math.log((N - cnt + 0.5) / (cnt + 0.5) + 1)
            for term, cnt in df.items()
        }

        print(f"[BM25] Indexed {N} docs | avgdl={self.avgdl:.1f} | vocab={len(self.idf)}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def score(self, query_tokens: List[str], doc_id: int) -> float:
        dl   = self.doc_lens[doc_id]
        tf_d = self.tf[doc_id]
        k1, b, avgdl = self.k1, self.b, self.avgdl
        s = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            f = tf_d.get(term, 0)
            numerator   = f * (k1 + 1)
            denominator = f + k1 * (1 - b + b * dl / avgdl)
            s += self.idf[term] * numerator / denominator
        return s

    def query(self, text: str, top_k: int = 20) -> List[Tuple[int, float]]:
        tokens = preprocess(text, stem=self.stem)
        if not tokens:
            return []

        # Only score docs that contain at least one query term (fast path)
        candidate_ids: set = set()
        for term in tokens:
            for doc_id in self.doc_ids:
                if self.tf[doc_id].get(term, 0) > 0:
                    candidate_ids.add(doc_id)

        if not candidate_ids:
            return []

        results = [(doc_id, self.score(tokens, doc_id))
                   for doc_id in candidate_ids]
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

    CORPUS_DIR = "data/docs"
    QUERY_CSV  = "data/public_test_queries.csv"
    ANSWER_CSV = "data/public_test_answers.csv"
    OUTPUT_CSV = "submissions/bm25_submission.csv"

    corpus  = load_corpus(CORPUS_DIR)
    queries = load_queries(QUERY_CSV)
    answers = load_answers(ANSWER_CSV)

    retriever = BM25Retriever(k1=1.5, b=0.75, stem=True)
    retriever.fit(corpus)

    results = retriever.retrieve_all(queries, top_k=20)

    print("\n=== BM25 Evaluation ===")
    evaluate(results, answers)

    os.makedirs("submissions", exist_ok=True)
    save_submission(results, OUTPUT_CSV)
