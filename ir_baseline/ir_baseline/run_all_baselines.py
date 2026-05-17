"""
Run and compare sparse IR baselines for the Cranfield contest.

Default paths match this repository:
  python ir_baseline/ir_baseline/run_all_baselines.py

The script writes one submission per model plus the selected final
submission file, defaulting to submissions/nlp_submission.csv.
"""

import argparse
import csv
import os
import sys
import time

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_corpus, load_queries, load_answers, save_submission
from utils.evaluate    import evaluate, evaluate_per_query
from baselines.baseline1_tfidf    import TFIDFRetriever
from baselines.baseline2_bm25     import BM25Retriever
from baselines.baseline3_bm25_prf import PRFRetriever
from baselines.baseline4_bm25_ngram import EnhancedBM25Retriever


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def slugify(name):
    return (
        name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
    )


def parse_k_values(text):
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def save_per_query_report(report, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id",
            "precision",
            "recall",
            "f1",
            "relevant_ranks",
            "predicted_docIDs",
        ])
        for qid in sorted(report):
            row = report[qid]
            ranks = " ".join(
                f"{doc}:{rank if rank is not None else 'NA'}"
                for doc, rank in row["relevant_ranks"].items()
            )
            writer.writerow([
                qid,
                f"{row['precision']:.6f}",
                f"{row['recall']:.6f}",
                f"{row['f1']:.6f}",
                ranks,
                " ".join(map(str, row["predicted"])),
            ])


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
        report = evaluate_per_query(results, answers)
        report_path = os.path.join(out_dir, f"{slugify(name)}_per_query.csv")
        save_per_query_report(report, report_path)
        print(f"  Per-query report: {report_path}")

    fname = slugify(name) + "_submission.csv"
    save_submission(results, os.path.join(out_dir, fname))
    return {"name": name, "top_k": top_k, "metrics": metrics, "results": results}


def grid_search_top_k(retriever_cls, retriever_kwargs, corpus, queries, answers,
                      k_values=(2, 3, 5, 10, 15, 20, 30, 50)):
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
    parser.add_argument("--corpus",  default="data/Cranfield", help="Directory with *.txt docs")
    parser.add_argument("--queries", default="data/public_test_queries.csv", help="CSV with query_id,query")
    parser.add_argument("--answers", default=None,
                        help="CSV with query_id,relevant_docIDs; auto-used for public queries if missing")
    parser.add_argument("--output",  default="submissions", help="Output directory")
    parser.add_argument("--top_k",   type=int, default=None,
                        help="Fixed top_k (default: auto grid-search if --answers given)")
    parser.add_argument("--k_values", default="2,3,5,10,15,20,30,50",
                        help="Comma-separated top_k values for public-set diagnostics")
    parser.add_argument("--submission_name", default="nlp_submission.csv",
                        help="Final submission filename")
    parser.add_argument("--final_model", default="auto",
                        choices=("auto", "tfidf", "bm25", "bm25_prf", "enhanced_bm25"),
                        help="Model used for final nlp_submission.csv")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if not os.path.isdir(args.corpus):
        raise FileNotFoundError(f"Corpus directory not found: {args.corpus}")
    if not os.path.isfile(args.queries):
        raise FileNotFoundError(f"Query CSV not found: {args.queries}")

    # Load data
    print("Loading data ...")
    corpus  = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    answer_path = args.answers
    if answer_path is None and os.path.basename(args.queries) == "public_test_queries.csv":
        answer_path = os.path.join(os.path.dirname(args.queries), "public_test_answers.csv")
    answers = load_answers(answer_path) if answer_path and os.path.isfile(answer_path) else None
    print(f"  Corpus : {len(corpus)} docs")
    print(f"  Queries: {len(queries)} queries")
    print(f"  Answers: {len(answers) if answers else 0} queries")
    k_values = parse_k_values(args.k_values)

    model_specs = [
        ("tfidf", "TF-IDF", TFIDFRetriever, {"stem": True}),
        ("bm25", "BM25", BM25Retriever, {"k1": 1.5, "b": 0.75, "stem": True}),
        (
            "bm25_prf",
            "BM25+PRF",
            PRFRetriever,
            {
                "k1": 1.5,
                "b": 0.75,
                "prf_top_docs": 5,
                "prf_top_terms": 10,
                "alpha": 1.0,
                "beta": 0.8,
                "stem": True,
            },
        ),
        (
            "enhanced_bm25",
            "Enhanced BM25",
            EnhancedBM25Retriever,
            {
                "k1": 1.5,
                "b": 0.75,
                "stem": True,
                "max_ngram": 3,
                "bigram_weight": 0.12,
                "trigram_weight": 0.18,
                "exact_phrase_boost": 0.1,
                "candidate_pool": 100,
            },
        ),
    ]

    runs = {}
    for idx, (key, name, cls, kwargs) in enumerate(model_specs, start=1):
        print(f"\n[{idx}/{len(model_specs)}] {name}")
        if answers and args.top_k is None:
            print("  Grid searching top_k for diagnostics ...")
            top_k = grid_search_top_k(cls, kwargs, corpus, queries, answers, k_values)
        else:
            top_k = args.top_k or 20

        run = run_baseline(name, cls(**kwargs), corpus, queries, answers, top_k, args.output)
        runs[key] = run

    if args.final_model == "auto":
        if answers:
            final_key = max(
                runs,
                key=lambda key: runs[key]["metrics"]["f1"] if runs[key]["metrics"] else -1.0,
            )
        else:
            final_key = "enhanced_bm25"
    else:
        final_key = args.final_model

    final_path = os.path.join(args.output, args.submission_name)
    save_submission(runs[final_key]["results"], final_path)
    print(f"\nFinal model : {runs[final_key]['name']}  |  top_k={runs[final_key]['top_k']}")
    print(f"Final file  : {final_path}")
    print(f"\nAll diagnostics/submissions saved to: {args.output}/")


if __name__ == "__main__":
    main()
