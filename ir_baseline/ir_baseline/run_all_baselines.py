"""
run_all_baselines.py
====================
Chạy và so sánh tất cả baselines trên public_test set.
Tự động tìm best top-K cho từng model.

Cách dùng:
  python run_all_baselines.py \
      --corpus  data/docs \
      --queries data/public_test_queries.csv \
      --answers data/public_test_answers.csv \
      --output  submissions/

  Nếu không có answers (vòng bí mật), bỏ --answers để chỉ tạo submission.
"""

import argparse
import os
import sys
import time

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_corpus, load_queries, load_answers, save_submission
from utils.evaluate    import evaluate
from baselines.baseline1_tfidf    import TFIDFRetriever
from baselines.baseline2_bm25     import BM25Retriever
from baselines.baseline3_bm25_prf import PRFRetriever


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def run_baseline(name, retriever, corpus, queries, answers, top_k, out_dir):
    t0 = time.time()
    retriever.fit(corpus)
    results = retriever.retrieve_all(queries, top_k=top_k)
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"  {name}  |  top_k={top_k}  |  {elapsed:.1f}s")
    print(f"{'='*50}")

    metrics = None
    if answers:
        metrics = evaluate(results, answers)

    fname = name.lower().replace(" ", "_").replace("+", "plus") + "_submission.csv"
    save_submission(results, os.path.join(out_dir, fname))
    return metrics


def grid_search_top_k(retriever_cls, retriever_kwargs, corpus, queries, answers,
                      k_values=(5, 10, 15, 20, 30, 50)):
    """Find best top_k on public answers via grid search."""
    best_k, best_f1 = k_values[0], -1.0
    retriever = retriever_cls(**retriever_kwargs)
    retriever.fit(corpus)
    for k in k_values:
        results = retriever.retrieve_all(queries, top_k=k)
        m = evaluate(results, answers, verbose=False)
        print(f"    top_k={k:3d}  →  F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")
        if m["f1"] > best_f1:
            best_f1, best_k = m["f1"], k
    print(f"  ★ Best top_k = {best_k}  (F1 = {best_f1:.4f})")
    return best_k


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",  required=True, help="Directory with *.txt docs")
    parser.add_argument("--queries", required=True, help="CSV with query_id,query")
    parser.add_argument("--answers", default=None,  help="CSV with query_id,relevant_docIDs")
    parser.add_argument("--output",  default="submissions", help="Output directory")
    parser.add_argument("--top_k",   type=int, default=None,
                        help="Fixed top_k (default: auto grid-search if --answers given)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load data
    print("Loading data ...")
    corpus  = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    answers = load_answers(args.answers) if args.answers else None
    print(f"  Corpus : {len(corpus)} docs")
    print(f"  Queries: {len(queries)} queries")

    # ----------------------------------------------------------------
    # Baseline 1: TF-IDF
    # ----------------------------------------------------------------
    print("\n[1/3] TF-IDF + Cosine Similarity")
    if answers and args.top_k is None:
        print("  Grid searching best top_k ...")
        best_k1 = grid_search_top_k(
            TFIDFRetriever, {"stem": True}, corpus, queries, answers
        )
    else:
        best_k1 = args.top_k or 20

    run_baseline("TF-IDF", TFIDFRetriever(stem=True),
                 corpus, queries, answers, best_k1, args.output)

    # ----------------------------------------------------------------
    # Baseline 2: BM25
    # ----------------------------------------------------------------
    print("\n[2/3] BM25 (Okapi BM25)")
    if answers and args.top_k is None:
        print("  Grid searching best top_k ...")
        best_k2 = grid_search_top_k(
            BM25Retriever, {"k1": 1.5, "b": 0.75, "stem": True},
            corpus, queries, answers
        )
    else:
        best_k2 = args.top_k or 20

    run_baseline("BM25", BM25Retriever(k1=1.5, b=0.75, stem=True),
                 corpus, queries, answers, best_k2, args.output)

    # ----------------------------------------------------------------
    # Baseline 3: BM25 + PRF
    # ----------------------------------------------------------------
    print("\n[3/3] BM25 + Pseudo-Relevance Feedback")
    if answers and args.top_k is None:
        print("  Grid searching best top_k ...")
        best_k3 = grid_search_top_k(
            PRFRetriever,
            {"k1": 1.5, "b": 0.75, "prf_top_docs": 5,
             "prf_top_terms": 10, "alpha": 1.0, "beta": 0.8, "stem": True},
            corpus, queries, answers
        )
    else:
        best_k3 = args.top_k or 20

    run_baseline("BM25+PRF", PRFRetriever(
                     k1=1.5, b=0.75, prf_top_docs=5, prf_top_terms=10,
                     alpha=1.0, beta=0.8, stem=True),
                 corpus, queries, answers, best_k3, args.output)

    print(f"\n✓ All submissions saved to: {args.output}/")


if __name__ == "__main__":
    main()
