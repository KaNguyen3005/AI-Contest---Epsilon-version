"""
Baseline 4 - BM25 + n-gram and exact phrase boost.

This retriever stays within the contest constraint: no pretrained model, no
external dependency. It extends BM25 with phrase-aware sparse features because
Cranfield queries often contain technical phrases such as "boundary layer" or
"hypersonic flow".
"""

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from baselines.baseline2_bm25 import BM25Retriever
from utils.preprocessing import (
    make_ngrams,
    preprocess,
    remove_stopwords,
    simple_tokenize,
)


class EnhancedBM25Retriever(BM25Retriever):
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stem: bool = True,
        max_ngram: int = 3,
        bigram_weight: float = 0.12,
        trigram_weight: float = 0.18,
        exact_phrase_boost: float = 0.1,
        candidate_pool: int = 100,
    ):
        super().__init__(k1=k1, b=b, stem=stem)
        self.max_ngram = max_ngram
        self.bigram_weight = bigram_weight
        self.trigram_weight = trigram_weight
        self.exact_phrase_boost = exact_phrase_boost
        self.candidate_pool = candidate_pool
        self.raw_docs: Dict[int, str] = {}
        self.doc_ngrams: Dict[int, Set[str]] = {}
        self.ngram_idf: Dict[str, float] = {}

    def fit(self, corpus: Dict[int, str]):
        self.raw_docs = {
            doc_id: " ".join(simple_tokenize(text))
            for doc_id, text in corpus.items()
        }
        super().fit(corpus)
        self._build_ngram_stats(corpus)

    def _build_ngram_stats(self, corpus: Dict[int, str]):
        df: Dict[str, int] = defaultdict(int)
        self.doc_ngrams = {}
        for doc_id, text in corpus.items():
            tokens = preprocess(text, stem=self.stem)
            grams: Set[str] = set()
            if self.max_ngram >= 2:
                grams.update(make_ngrams(tokens, 2))
            if self.max_ngram >= 3:
                grams.update(make_ngrams(tokens, 3))
            self.doc_ngrams[doc_id] = grams
            for gram in grams:
                df[gram] += 1

        n_docs = len(corpus)
        self.ngram_idf = {
            gram: math.log((n_docs - cnt + 0.5) / (cnt + 0.5) + 1.0)
            for gram, cnt in df.items()
        }

    def _query_ngrams(self, text: str) -> Dict[str, float]:
        tokens = preprocess(text, stem=self.stem)
        weights: Dict[str, float] = defaultdict(float)

        if self.max_ngram >= 2:
            for token in make_ngrams(tokens, 2):
                weights[token] += self.bigram_weight

        if self.max_ngram >= 3:
            for token in make_ngrams(tokens, 3):
                weights[token] += self.trigram_weight

        return weights

    def _raw_phrases(self, text: str) -> List[str]:
        raw_tokens = remove_stopwords(simple_tokenize(text))
        phrases = []
        for n in (3, 2):
            for grams in zip(*(raw_tokens[i:] for i in range(n))):
                phrase = " ".join(grams)
                if len(phrase) >= 7:
                    phrases.append(phrase)
        return phrases

    def _phrase_bonus(self, text: str, doc_id: int) -> float:
        if self.exact_phrase_boost <= 0:
            return 0.0
        doc_text = self.raw_docs.get(doc_id, "")
        bonus = 0.0
        for phrase in self._raw_phrases(text):
            if phrase in doc_text:
                bonus += self.exact_phrase_boost
        return bonus

    def _ngram_bonus(self, query_ngrams: Dict[str, float], doc_id: int) -> float:
        doc_grams = self.doc_ngrams.get(doc_id, set())
        bonus = 0.0
        for gram, weight in query_ngrams.items():
            if gram in doc_grams:
                bonus += weight * self.ngram_idf.get(gram, 0.0)
        return bonus

    def query(self, text: str, top_k: int = 20) -> List[Tuple[int, float]]:
        base_ranked = super().query(text, top_k=max(top_k, self.candidate_pool))
        if not base_ranked:
            return []

        query_ngrams = self._query_ngrams(text)
        results = []
        for doc_id, base_score in base_ranked:
            score = base_score
            score += self._ngram_bonus(query_ngrams, doc_id)
            score += self._phrase_bonus(text, doc_id)
            results.append((doc_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from utils.data_loader import load_answers, load_corpus, load_queries, save_submission
    from utils.evaluate import evaluate

    CORPUS_DIR = "data/Cranfield"
    QUERY_CSV = "data/public_test_queries.csv"
    ANSWER_CSV = "data/public_test_answers.csv"
    OUTPUT_CSV = "submissions/enhanced_bm25_submission.csv"

    corpus = load_corpus(CORPUS_DIR)
    queries = load_queries(QUERY_CSV)
    answers = load_answers(ANSWER_CSV)

    retriever = EnhancedBM25Retriever()
    retriever.fit(corpus)
    results = retriever.retrieve_all(queries, top_k=20)

    print("\n=== Enhanced BM25 Evaluation ===")
    evaluate(results, answers)

    os.makedirs("submissions", exist_ok=True)
    save_submission(results, OUTPUT_CSV)
